import React, { useState } from 'react';
import { Search, Brain, Compass, Loader2, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { toast } from '@/hooks/use-toast';

interface SearchResult {
  content: string;
  score: number;
  source: string;
  metadata?: Record<string, any>;
}

interface SearchResponse {
  success: boolean;
  message: string;
  results: SearchResult[];
  action: string;
  query: string;
}

interface SearchInterfaceProps {
  onSearch: (results: SearchResponse) => void;
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeAction, setActiveAction] = useState<'search' | 'explore' | 'think'>('search');

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: "Empty Query",
        description: "Please enter a search query",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          action: activeAction,
          threshold: 0.35,
        }),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const result: SearchResponse = await response.json();
      onSearch(result);

      if (result.success) {
        toast({
          title: "Search Complete",
          description: `Found ${result.results.length} relevant results`,
        });
      } else {
        toast({
          title: "Search Issue",
          description: result.message,
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Search error:', error);
      toast({
        title: "Search Error",
        description: error instanceof Error ? error.message : "Search failed",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const actionButtons = [
    {
      id: 'search' as const,
      label: 'Search',
      icon: Search,
      description: 'Find documents matching your query',
      color: 'primary',
    },
    {
      id: 'explore' as const,
      label: 'Explore',
      icon: Compass,
      description: 'Deep exploration with web search',
      color: 'secondary',
    },
    {
      id: 'think' as const,
      label: 'Think',
      icon: Brain,
      description: 'Generate insights and analysis',
      color: 'accent',
    },
  ];

  return (
    <Card className="glass">
      <div className="p-6 space-y-6">
        <div>
          <h2 className="text-2xl font-bold gradient-text mb-2">
            Smart Document Search
          </h2>
          <p className="text-muted-foreground">
            Ask questions, explore topics, or find specific information in your documents
          </p>
        </div>

        {/* Action Selection */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {actionButtons.map((action) => {
            const Icon = action.icon;
            const isActive = activeAction === action.id;
            return (
              <Button
                key={action.id}
                variant={isActive ? action.color as any : "outline"}
                onClick={() => setActiveAction(action.id)}
                className={`p-4 h-auto flex-col space-y-2 ${
                  isActive ? 'ring-2 ring-current' : ''
                }`}
              >
                <Icon className="w-5 h-5" />
                <div className="text-center">
                  <div className="font-medium">{action.label}</div>
                  <div className="text-xs opacity-80 leading-tight">
                    {action.description}
                  </div>
                </div>
              </Button>
            );
          })}
        </div>

        {/* Search Input */}
        <div className="space-y-3">
          <div className="relative">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="What would you like to know about your documents?"
              className="pr-12 h-12 text-base bg-muted/30 border-primary/20 focus:border-primary"
              disabled={loading}
            />
            <Button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              variant="ghost"
              size="icon"
              className="absolute right-1 top-1 h-10 w-10"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>

          <div className="flex items-center space-x-2 text-sm text-muted-foreground">
            <span>Active mode:</span>
            <Badge variant="outline" className="border-primary/40">
              {actionButtons.find(a => a.id === activeAction)?.label}
            </Badge>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2">
          {[
            "Summarize all documents",
            "Find key insights",
            "Extract action items",
            "Compare documents",
          ].map((suggestion, index) => (
            <Button
              key={index}
              variant="ghost"
              size="sm"
              onClick={() => setQuery(suggestion)}
              className="text-xs hover:bg-primary/10"
            >
              {suggestion}
            </Button>
          ))}
        </div>
      </div>
    </Card>
  );
};

export default SearchInterface;