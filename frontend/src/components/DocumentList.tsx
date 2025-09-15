import React, { useState, useEffect } from 'react';
import { FileText, Image, Calendar, Hash, Loader2, RefreshCw, Trash2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { toast } from '@/hooks/use-toast';

interface DocumentInfo {
  name: string;
  content_length: number;
  upload_date: string;
}

interface DocumentsData {
  total_documents: number;
  documents: DocumentInfo[];
  clusters: number;
  timestamp: string;
}

interface DocumentListProps {
  refreshTrigger: number;
}

const DocumentList: React.FC<DocumentListProps> = ({ refreshTrigger }) => {
  const [documentsData, setDocumentsData] = useState<DocumentsData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/documents');
      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.statusText}`);
      }
      const data: DocumentsData = await response.json();
      setDocumentsData(data);
    } catch (error) {
      console.error('Error fetching documents:', error);
      toast({
        title: "Error",
        description: "Failed to load documents",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReload = async () => {
    try {
      const response = await fetch('http://localhost:8000/documents/reload', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(`Failed to reload documents: ${response.statusText}`);
      }
      toast({
        title: "Success",
        description: "Documents reloaded successfully",
      });
      fetchDocuments();
    } catch (error) {
      console.error('Error reloading documents:', error);
      toast({
        title: "Error",
        description: "Failed to reload documents",
        variant: "destructive",
      });
    }
  };

  const handleReset = async () => {
    if (!confirm('Are you sure? This will delete all documents and reset the system.')) {
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/reset', {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error(`Failed to reset system: ${response.statusText}`);
      }
      toast({
        title: "System Reset",
        description: "All documents have been removed",
      });
      fetchDocuments();
    } catch (error) {
      console.error('Error resetting system:', error);
      toast({
        title: "Error",
        description: "Failed to reset system",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [refreshTrigger]);

  const getFileIcon = (fileName: string) => {
    const ext = fileName.toLowerCase().split('.').pop();
    if (ext === 'jpg' || ext === 'jpeg' || ext === 'png') {
      return <Image className="w-4 h-4 text-accent" />;
    }
    return <FileText className="w-4 h-4 text-primary" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <Card className="glass">
        <div className="p-8 text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
          <h3 className="text-lg font-medium mb-2">Loading Documents</h3>
          <p className="text-muted-foreground">Fetching your document library...</p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="glass">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold gradient-text">Document Library</h2>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm" onClick={fetchDocuments}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
              <Button variant="outline" size="sm" onClick={handleReload}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Reload
              </Button>
              <Button variant="destructive" size="sm" onClick={handleReset}>
                <Trash2 className="w-4 h-4 mr-2" />
                Reset
              </Button>
            </div>
          </div>

          {documentsData && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="flex items-center space-x-3 p-4 bg-primary/10 rounded-lg">
                <FileText className="w-8 h-8 text-primary" />
                <div>
                  <p className="text-2xl font-bold text-primary">
                    {documentsData.total_documents}
                  </p>
                  <p className="text-sm text-muted-foreground">Total Documents</p>
                </div>
              </div>

              <div className="flex items-center space-x-3 p-4 bg-secondary/10 rounded-lg">
                <Hash className="w-8 h-8 text-secondary" />
                <div>
                  <p className="text-2xl font-bold text-secondary">
                    {documentsData.clusters}
                  </p>
                  <p className="text-sm text-muted-foreground">Clusters</p>
                </div>
              </div>

              <div className="flex items-center space-x-3 p-4 bg-accent/10 rounded-lg">
                <Calendar className="w-8 h-8 text-accent" />
                <div>
                  <p className="text-sm font-medium text-accent">
                    Last Updated
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {formatDate(documentsData.timestamp)}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {documentsData && documentsData.documents.length === 0 ? (
        <Card className="glass">
          <div className="p-8 text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-muted to-muted/50 rounded-full flex items-center justify-center mb-4 mx-auto">
              <FileText className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-medium text-muted-foreground mb-2">
              No Documents Found
            </h3>
            <p className="text-sm text-muted-foreground">
              Upload your first document to get started with AI-powered search
            </p>
          </div>
        </Card>
      ) : (
        documentsData && (
          <div className="space-y-3">
            {documentsData.documents.map((doc, index) => (
              <Card key={index} className="glass hover:shadow-lg transition-all duration-300">
                <div className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3 flex-1 min-w-0">
                      {getFileIcon(doc.name)}
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium truncate" title={doc.name}>
                          {doc.name}
                        </h3>
                        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                          <span>{formatFileSize(doc.content_length)}</span>
                          <span>{formatDate(doc.upload_date)}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="text-xs">
                        {doc.name.split('.').pop()?.toUpperCase()}
                      </Badge>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )
      )}
    </div>
  );
};

export default DocumentList;