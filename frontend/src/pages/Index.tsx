import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import DocumentUpload from '@/components/DocumentUpload';
import SearchInterface from '@/components/SearchInterface';
import SearchResults from '@/components/SearchResults';
import DocumentList from '@/components/DocumentList';
import { Button } from '@/components/ui/button';
import { FileText, Search, Upload, List, Github, Zap } from 'lucide-react';

interface SearchResponse {
  success: boolean;
  message: string;
  results: any[];
  action: string;
  query: string;
  timestamp?: string;
}

const Index = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [activeTab, setActiveTab] = useState('search');

  const handleUploadComplete = () => {
    setRefreshTrigger(prev => prev + 1);
    setActiveTab('search'); // Switch to search tab after upload
  };

  const handleSearchResults = (results: SearchResponse) => {
    setSearchResults(results);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      {/* Header */}
      <header className="border-b border-border/30 bg-card/30 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">
                  Smart Document Search
                </h1>
                <p className="text-sm text-muted-foreground">
                  AI-Powered Knowledge Discovery
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm" asChild>
                <a 
                  href="https://github.com" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2"
                >
                  <Github className="w-4 h-4" />
                  <span className="hidden sm:inline">GitHub</span>
                </a>
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="grid w-full grid-cols-3 lg:w-[400px] mx-auto bg-muted/50 backdrop-blur-sm">
              <TabsTrigger value="search" className="flex items-center space-x-2">
                <Search className="w-4 h-4" />
                <span>Search</span>
              </TabsTrigger>
              <TabsTrigger value="upload" className="flex items-center space-x-2">
                <Upload className="w-4 h-4" />
                <span>Upload</span>
              </TabsTrigger>
              <TabsTrigger value="documents" className="flex items-center space-x-2">
                <List className="w-4 h-4" />
                <span>Library</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="search" className="space-y-8">
              <SearchInterface onSearch={handleSearchResults} />
              <SearchResults searchResponse={searchResults} />
            </TabsContent>

            <TabsContent value="upload" className="space-y-8">
              <DocumentUpload onUploadComplete={handleUploadComplete} />
            </TabsContent>

            <TabsContent value="documents" className="space-y-8">
              <DocumentList refreshTrigger={refreshTrigger} />
            </TabsContent>
          </Tabs>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/30 bg-card/20 backdrop-blur-sm mt-16">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center text-muted-foreground">
            <p className="text-sm">
              Built with{' '}
              <span className="gradient-text font-medium">FastAPI</span> and{' '}
              <span className="gradient-text font-medium">React</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
