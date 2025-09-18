import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Image, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';

// Interface to match the nested message object from the API
interface UploadMessage {
  text_inserted?: boolean;
  images_inserted?: number;
  message: string;
}

// Updated interface to match the API response structure
interface UploadedFile {
  filename: string; // FIX: Changed from 'name' to 'filename'
  status: 'success' | 'error';
  message: string | UploadMessage; // FIX: Can be a string or the object type
}

interface DocumentUploadProps {
  onUploadComplete: () => void;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadedFile[]>([]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setUploading(true);
    setUploadResults([]);

    try {
      const formData = new FormData();
      acceptedFiles.forEach((file) => {
        formData.append('files', file);
      });

      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`Upload failed: ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();
      setUploadResults(result.uploaded_files || []);
      
      const successCount = result.uploaded_files?.filter((f: UploadedFile) => f.status === 'success').length || 0;
      
      if (successCount > 0) {
        toast({
          title: "Upload Successful",
          description: `Successfully uploaded ${successCount}/${acceptedFiles.length} files`,
        });
        onUploadComplete();
      } else {
        toast({
          title: "Upload Failed",
          description: result.message || "No files were uploaded successfully",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast({
        title: "Upload Error",
        description: error instanceof Error ? error.message : "Upload failed due to a network or server error.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'text/plain': ['.txt'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
    },
    multiple: true,
  });

  const getFileIcon = (fileName: string) => {
    // FIX: Add a check to prevent errors if fileName is somehow still undefined
    if (!fileName) return <FileText className="w-4 h-4" />;
    
    const ext = fileName.toLowerCase().split('.').pop();
    if (ext === 'pdf' || ext === 'docx' || ext === 'pptx' || ext === 'txt') {
      return <FileText className="w-4 h-4" />;
    }
    if (ext === 'jpg' || ext === 'jpeg' || ext === 'png') {
      return <Image className="w-4 h-4" />;
    }
    return <FileText className="w-4 h-4" />;
  };
  
  // Helper to render the message correctly
  const renderUploadMessage = (message: string | UploadMessage) => {
    if (typeof message === 'string') {
      return message;
    }
    return message.message || 'Processing complete.';
  };

  return (
    <div className="space-y-6">
      <Card className="glass border-2 border-dashed border-primary/30 hover:border-primary/60 transition-all duration-300">
        <div
          {...getRootProps()}
          className={`p-8 text-center cursor-pointer transition-all duration-300 ${
            isDragActive ? 'bg-primary/10 scale-105' : 'hover:bg-primary/5'
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center space-y-4">
            <div className="w-16 h-16 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center mb-4">
              <Upload className="w-8 h-8 text-primary-foreground" />
            </div>
            
            {uploading ? (
              <div className="flex items-center space-x-2">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="text-lg font-medium">Uploading files...</span>
              </div>
            ) : (
              <>
                <h3 className="text-xl font-semibold gradient-text">
                  {isDragActive ? 'Drop files here' : 'Upload Documents'}
                </h3>
                <p className="text-muted-foreground max-w-md">
                  Drag & drop your documents here, or click to browse. 
                  Supports PDF, DOCX, PPTX, TXT, JPG, PNG files.
                </p>
                <Button variant="outline" size="lg" className="mt-4">
                  Browse Files
                </Button>
              </>
            )}
          </div>
        </div>
      </Card>

      {uploadResults.length > 0 && (
        <Card className="glass">
          <div className="p-6">
            <h4 className="text-lg font-semibold mb-4">Upload Results</h4>
            <div className="space-y-3">
              {uploadResults.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-3 p-3 rounded-lg bg-muted/30"
                >
                  {/* FIX: Use file.filename */}
                  {getFileIcon(file.filename)}
                  <span className="flex-1 font-medium truncate">{file.filename}</span>
                  {file.status === 'success' ? (
                    <CheckCircle className="w-5 h-5 text-success" />
                  ) : (
                    <XCircle className="w-5 h-5 text-destructive" />
                  )}
                  <span
                    className={`text-sm ${
                      file.status === 'success' ? 'text-success' : 'text-destructive'
                    }`}
                  >
                    {/* FIX: Use helper function to render message */}
                    {renderUploadMessage(file.message)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default DocumentUpload;