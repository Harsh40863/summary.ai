import React, { useMemo, useState } from 'react';
import { FileText, Star, Clock, ExternalLink, Search, MapPin, Zap } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

// Markdown rendering + sanitization
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';

/**
 * Unified Search / Explore / Think results component
 * - Default export React component (single-file)
 * - Uses Tailwind + shadcn/ui primitives (Card, Badge, Button)
 * - Accepts the same `SearchResponse` shape you provided
 * - Provides a top-level action badge and an inline filter to show only
 *   results whose `action_type` matches 'search' | 'explore' | 'think'
 */

interface SearchResultRaw {
  document_name?: string;
  best_sentence?: string;
  summary?: string;
  score?: number;
  action_type?: string; // per-result action type
  metadata?: Record<string, any>;
}

interface SearchResponse {
  success: boolean;
  message?: string;
  results?: SearchResultRaw[];
  action?: string; // top-level action
  query?: string;
  timestamp?: string;
}

interface Props {
  searchResponse: SearchResponse | null;
}

const ACTIONS = ['all', 'search', 'explore', 'think'] as const;
type ActionFilter = typeof ACTIONS[number];

const getActionColor = (action?: string) => {
  switch ((action || '').toLowerCase()) {
    case 'search':
      return 'primary';
    case 'explore':
      return 'secondary';
    case 'think':
      return 'accent';
    default:
      return 'outline';
  }
};

const safeGetContent = (r: SearchResultRaw) => {
  // prefer the most informative field we have: summary -> web_content -> best_sentence
  return (
    r.summary?.trim() ||
    // web_content can be large and contain markdown/html — keep as plain text here
    // rendering will use a markdown/HTML-safe renderer below
    (r as any).web_content?.trim() ||
    r.best_sentence?.trim() ||
    '[no excerpt available]'
  );
};

const formatScore = (raw?: number) => {
  if (raw === undefined || raw === null || Number.isNaN(raw)) return '';
  if (raw >= 0 && raw <= 1) return (raw * 100).toFixed(0) + '%';
  if (raw > 1 && raw <= 1000) return Math.round(raw) + '%';
  return (raw / 10).toFixed(2) + '%';
};

const getScoreColorClass = (raw?: number) => {
  if (raw === undefined || raw === null || Number.isNaN(raw)) return 'text-muted-foreground';
  let s = raw;
  if (raw > 1 && raw <= 1000) s = raw / 100; // normalize percent-like to 0..1
  if (s >= 0.8) return 'text-success';
  if (s >= 0.6) return 'text-warning';
  return 'text-muted-foreground';
};

const ActionIcon: React.FC<{ action?: string }> = ({ action }) => {
  switch ((action || '').toLowerCase()) {
    case 'search':
      return <Search className="w-4 h-4" />;
    case 'explore':
      return <MapPin className="w-4 h-4" />;
    case 'think':
      return <Zap className="w-4 h-4" />;
    default:
      return <FileText className="w-4 h-4" />;
  }
};

/**
 * ResultCard - handles rendering a single result and formats `web_content`
 * nicely using a markdown renderer with sanitization and a simple
 * "show more" toggle for long content.
 */
const ResultCard: React.FC<{ r: SearchResultRaw; idx: number; topAction?: string }> = ({ r, idx, topAction }) => {
  const [expanded, setExpanded] = useState(false);
  const contentRaw = safeGetContent(r);

  const actionType = (r.action_type || r.metadata?.action_type || topAction || '').toLowerCase();

  // heuristics: decide whether content looks like markdown/html so we render it as rich text
  const looksLikeMarkdownOrHtml = (text: string) => {
    const mdHints = ['#', '*', '-', '`', '```', '>', '[', '](', '<p', '<div', '<br', '<img'];
    return mdHints.some(h => text.includes(h));
  };

  const shouldRenderRich = looksLikeMarkdownOrHtml(contentRaw);

  const shortPreview = contentRaw.length > 600 ? contentRaw.slice(0, 600).trim() + '…' : contentRaw;

  const scoreLabel = formatScore(r.score);
  const scoreClass = getScoreColorClass(r.score);

  return (
    <Card className="glass hover:shadow-lg transition-all duration-300">
      <div className="p-6">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-2">
            <FileText className="w-5 h-5 text-primary" />
            <h3 className="font-medium text-lg truncate max-w-md">{r.document_name || `Result ${idx + 1}`}</h3>

            <Badge variant={getActionColor(actionType) as any} className="ml-2 lowercase">
              {actionType || 'unknown'}
            </Badge>
          </div>

          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              <Star className={`w-4 h-4 ${scoreClass}`} />
              <span className={`text-sm font-medium ${scoreClass}`}>{scoreLabel}</span>
            </div>

            {r.metadata?.url && (
              <Button variant="ghost" size="sm" asChild>
                <a href={String(r.metadata.url)} target="_blank" rel="noopener noreferrer" className="flex items-center space-x-1">
                  <ExternalLink className="w-3 h-3" />
                  <span className="text-xs">Open</span>
                </a>
              </Button>
            )}
          </div>
        </div>

        <div className="prose prose-sm max-w-none">
          {/* Render either rich markdown/HTML or plain text depending on heuristics */}
          {shouldRenderRich ? (
            <div>
              {/* Show short preview when collapsed to avoid huge cards */}
              {!expanded ? (
                <div className="whitespace-pre-line overflow-hidden">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw, rehypeSanitize] as any}
                  >
                    {shortPreview}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="whitespace-pre-line">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw, rehypeSanitize] as any}
                  >
                    {contentRaw}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          ) : (
            <p className="text-foreground leading-relaxed whitespace-pre-line">{expanded ? contentRaw : shortPreview}</p>
          )}

          {/* Show toggle when content is long */}
          {contentRaw.length > 600 && (
            <div className="mt-3">
              <Button size="sm" variant="ghost" onClick={() => setExpanded(v => !v)}>
                {expanded ? 'Show less' : 'Show more'}
              </Button>
            </div>
          )}
        </div>

        {r.metadata && Object.keys(r.metadata).length > 0 && (
          <div className="mt-4 pt-4 border-t border-border/30">
            <div className="flex flex-wrap gap-2">
              {Object.entries(r.metadata)
                .filter(([key]) => key !== 'url')
                .map(([key, value]) => (
                  <Badge key={key} variant="outline" className="text-xs">
                    {key}: {String(value)}
                  </Badge>
                ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

const SearchResultsUnified: React.FC<Props> = ({ searchResponse }) => {
  const [filter, setFilter] = useState<ActionFilter>('all');

  const results = searchResponse?.results ?? [];

  const filteredResults = useMemo(() => {
    if (!results || results.length === 0) return [];
    if (filter === 'all') return results;

    return results.filter(r => {
      // normalize action checks to include per-result action_type, fallback to metadata or top-level action
      const action = (r.action_type || r.metadata?.action_type || searchResponse?.action || '').toString().toLowerCase();
      return action === filter;
    });
  }, [results, filter, searchResponse]);

  if (!searchResponse) {
    return (
      <Card className="glass">
        <div className="p-8 text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-muted to-muted/50 rounded-full flex items-center justify-center mb-4 mx-auto">
            <FileText className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium text-muted-foreground mb-2">No Search Results</h3>
          <p className="text-sm text-muted-foreground">Start by uploading documents and searching for information</p>
        </div>
      </Card>
    );
  }

  if (!searchResponse.success) {
    return (
      <Card className="glass border-destructive/20">
        <div className="p-6">
          <h3 className="text-lg font-medium text-destructive mb-2">Search Failed</h3>
          <p className="text-muted-foreground">{searchResponse.message}</p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="glass">
        <div className="p-6">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-semibold">Search Results</h2>
              <Badge variant={getActionColor(searchResponse.action) as any} className="flex items-center gap-2">
                <ActionIcon action={searchResponse.action} />
                <span className="uppercase text-xs">{(searchResponse.action || 'result').toUpperCase()}</span>
              </Badge>
            </div>

            <div className="flex items-center gap-2">
              <div className="text-sm text-muted-foreground mr-4">
                <strong>Query:</strong> "{searchResponse.query ?? ''}"
              </div>

              <div className="flex items-center space-x-2">
                {ACTIONS.map(a => (
                  <Button
                    key={a}
                    size="sm"
                    variant={filter === a ? 'default' : 'ghost'}
                    onClick={() => setFilter(a)}
                    className="capitalize"
                  >
                    {a}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">Found {results.length} results</div>
            {searchResponse.timestamp && (
              <div className="text-xs text-muted-foreground flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                {new Date(searchResponse.timestamp).toLocaleString()}
              </div>
            )}
          </div>
        </div>
      </Card>

      {searchResponse.message && (
        <Card className="glass">
          <div className="p-6">
            <h3 className="text-lg font-medium mb-2">Summary</h3>
            <div className="prose prose-sm max-w-none">
              <p className="whitespace-pre-line">{searchResponse.message}</p>
            </div>
          </div>
        </Card>
      )}

      {filteredResults.length === 0 ? (
        <Card className="glass">
          <div className="p-8 text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-muted to-muted/50 rounded-full flex items-center justify-center mb-4 mx-auto">
              <FileText className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-medium text-muted-foreground mb-2">No Results Found</h3>
            <p className="text-sm text-muted-foreground">Try adjusting your search query or selecting a different action filter</p>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {filteredResults.map((r, i) => (
            <ResultCard key={i} r={r} idx={i} topAction={searchResponse.action} />
          ))}
        </div>
      )}
    </div>
  );
};

export default SearchResultsUnified;
