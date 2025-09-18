import React from 'react';
import { FileText, Star, Clock, ExternalLink } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface SearchResultRaw {
  document_name?: string;
  best_sentence?: string;
  summary?: string;
  score?: number;
  action_type?: string;
  metadata?: Record<string, any>;
}

interface SearchResponse {
  success: boolean;
  message?: string;
  results?: SearchResultRaw[];
  action?: string;
  query?: string;
  timestamp?: string;
}

interface SearchResultsProps {
  searchResponse: SearchResponse | null;
}

const getActionColor = (action?: string) => {
  switch ((action || '').toLowerCase()) {
    case 'search': return 'primary';
    case 'explore': return 'secondary';
    case 'think': return 'accent';
    default: return 'outline';
  }
};

const safeGetContent = (r: SearchResultRaw) => {
  return r.summary?.trim() || r.best_sentence?.trim() || '[no excerpt available]';
};

// Friendly score formatter tailored for the shape you provided:
// - If score is in [0..1] treat as similarity -> show percent
// - If score is in (1..1000] treat value as percent-like already (100 == 100%)
// - If extremely large, scale down
const formatScore = (raw?: number) => {
  if (raw === undefined || raw === null || Number.isNaN(raw)) return '';
  if (raw >= 0 && raw <= 1) return (raw * 100).toFixed(0) + '%';
  if (raw > 1 && raw <= 1000) return Math.round(raw) + '%';
  return (raw / 10).toFixed(2) + '%';
};

const getScoreColorClass = (raw?: number) => {
  if (raw === undefined || raw === null || Number.isNaN(raw)) return 'text-muted-foreground';
  let s = raw;
  if (raw > 1 && raw <= 1000) s = raw / 100; // map percent-like to 0..1 for color decision
  if (s >= 0.8) return 'text-success';
  if (s >= 0.6) return 'text-warning';
  return 'text-muted-foreground';
};

const SearchResults: React.FC<SearchResultsProps> = ({ searchResponse }) => {
  if (!searchResponse) {
    return (
      <Card className="glass">
        <div className="p-8 text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-muted to-muted/50 rounded-full flex items-center justify-center mb-4 mx-auto">
            <FileText className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium text-muted-foreground mb-2">
            No Search Results
          </h3>
          <p className="text-sm text-muted-foreground">
            Start by uploading documents and searching for information
          </p>
        </div>
      </Card>
    );
  }

  if (!searchResponse.success) {
    return (
      <Card className="glass border-destructive/20">
        <div className="p-6">
          <h3 className="text-lg font-medium text-destructive mb-2">
            Search Failed
          </h3>
          <p className="text-muted-foreground">{searchResponse.message}</p>
        </div>
      </Card>
    );
  }

  const results = searchResponse.results ?? [];

  return (
    <div className="space-y-6">
      <Card className="glass">
        <div className="p-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-semibold">Search Results</h2>
            <Badge variant={getActionColor(searchResponse.action) as any}>
              {(searchResponse.action || 'RESULT').toUpperCase()}
            </Badge>
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              <strong>Query:</strong> "{searchResponse.query ?? ''}"
            </p>
            <p className="text-sm text-muted-foreground">
              Found {results.length} results
            </p>
            {searchResponse.timestamp && (
              <p className="text-xs text-muted-foreground flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                {new Date(searchResponse.timestamp).toLocaleString()}
              </p>
            )}
          </div>
        </div>
      </Card>

      {/* Show backend message (if present) */}
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

      {results.length === 0 ? (
        <Card className="glass">
          <div className="p-8 text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-muted to-muted/50 rounded-full flex items-center justify-center mb-4 mx-auto">
              <FileText className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-medium text-muted-foreground mb-2">
              No Results Found
            </h3>
            <p className="text-sm text-muted-foreground">
              Try adjusting your search query or uploading more documents
            </p>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {results.map((r, i) => {
            const content = safeGetContent(r);
            const scoreLabel = formatScore(r.score);
            const scoreClass = getScoreColorClass(r.score);

            return (
              <Card key={i} className="glass hover:shadow-lg transition-all duration-300">
                <div className="p-6">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <FileText className="w-5 h-5 text-primary" />
                      <h3 className="font-medium text-lg truncate max-w-md">
                        {r.document_name || `Result ${i + 1}`}
                      </h3>
                    </div>

                    <div className="flex items-center space-x-2">
                      <div className="flex items-center space-x-1">
                        <Star className={`w-4 h-4 ${scoreClass}`} />
                        <span className={`text-sm font-medium ${scoreClass}`}>
                          {scoreLabel}
                        </span>
                      </div>

                      {r.metadata?.url && (
                        <Button variant="ghost" size="sm" asChild>
                          <a
                            href={String(r.metadata.url)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center space-x-1"
                          >
                            <ExternalLink className="w-3 h-3" />
                            <span className="text-xs">Open</span>
                          </a>
                        </Button>
                      )}
                    </div>
                  </div>

                  <div className="prose prose-sm max-w-none">
                    <p className="text-foreground leading-relaxed">
                      {content}
                    </p>
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
          })}
        </div>
      )}
    </div>
  );
};

export default SearchResults;
