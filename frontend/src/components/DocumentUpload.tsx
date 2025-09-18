import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Image, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';

interface UploadedFile {
  name: string;
  status: 'success' | 'error';
  message: string;
}

interface DocumentUploadProps {
  onUploadComplete: () => void;
}

const formatServerMessage = (msg: any): string => {
  if (!msg && msg !== '') return '';
  if (typeof msg === 'string') return msg;
  if (typeof msg === 'object') {
    const parts: string[] = [];
    if (typeof msg.message === 'string') parts.push(msg.message);
    if (typeof msg.text_inserted === 'boolean') parts.push(`Text inserted: ${msg.text_inserted}`);
    if (typeof msg.images_inserted === 'number') parts.push(`${msg.images_inserted} image(s) inserted`);
    if (parts.length === 0) return JSON.stringify(msg);
    return parts.join('. ');
  }
  return String(msg);
};

const MAX_FILE_BYTES = 50 * 1024 * 1024;

const DocumentUpload: React.FC<DocumentUploadProps> = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadedFile[]>([]);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      setUploading(true);
      setUploadResults([]);

      const rejected: string[] = [];
      const tooLarge: string[] = [];

      acceptedFiles.forEach((f) => {
        console.log('Dropped file:', f.name, f.type, f.size);
        if (f.size > MAX_FILE_BYTES) tooLarge.push(f.name);
        if (!['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'text/plain', 'image/jpeg', 'image/png'].includes(f.type) && !f.name.toLowerCase().endsWith('.pdf')) {
          rejected.push(`${f.name} (${f.type || 'unknown'})`);
        }
      });

      if (tooLarge.length > 0) {
        toast({
          title: 'File too large',
          description: `${tooLarge.join(', ')} exceeds ${MAX_FILE_BYTES / (1024 * 1024)} MB`,
          variant: 'destructive',
        });
        setUploading(false);
        return;
      }

      if (rejected.length > 0) {
        toast({
          title: 'Unsupported file type',
          description: `Rejected: ${rejected.join(', ')}`,
          variant: 'destructive',
        });
        setUploading(false);
        return;
      }

      try {
        const formData = new FormData();
        acceptedFiles.forEach((file) => {
          formData.append('files', file);
          formData.append('file', file);
          formData.append('files[]', file);
        });

        const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData,
        });

        let respText = '';
        let respJson: any = null;
        const contentType = response.headers.get('content-type') || '';
        try {
          if (contentType.includes('application/json')) {
            respJson = await response.json();
            respText = JSON.stringify(respJson);
          } else {
            respText = await response.text();
          }
        } catch (e) {
          respText = await response.text().catch(() => '');
        }

        if (!response.ok) {
          console.error('Upload failed', response.status, respText);
          toast({
            title: 'Upload failed',
            description: respText || `Status ${response.status}`,
            variant: 'destructive',
          });
          setUploadResults(
            acceptedFiles.map((f) => ({ name: f.name, status: 'error' as const, message: respText || `Status ${response.status}` }))
          );
          setUploading(false);
          return;
        }

        const result = respJson ?? {};
        const normalized: UploadedFile[] = (result?.uploaded_files ?? []).map((f: any) => ({
          name: typeof f?.filename === 'string' ? f.filename : (typeof f?.name === 'string' ? f.name : ''),
          status: f?.status === 'success' ? 'success' : 'error',
          message: formatServerMessage(f?.message),
        }));

        if (!Array.isArray(result?.uploaded_files) || normalized.length === 0) {
          const fallback = acceptedFiles.map((f) => ({ name: f.name, status: 'success' as const, message: 'Uploaded' }));
          setUploadResults(fallback);
        } else {
          setUploadResults(normalized);
        }

        const successCount = normalized.filter((f) => f.status === 'success').length;
        if (successCount > 0) {
          toast({ title: 'Upload Successful', description: `Successfully uploaded ${successCount}/${acceptedFiles.length} files` });
          onUploadComplete();
        } else {
          toast({ title: 'Upload Completed with Errors', description: 'Check results below', variant: 'destructive' });
        }
      } catch (error: any) {
        console.error('Upload error:', error);
        toast({
          title: 'Upload Error',
          description: error?.message ?? String(error),
          variant: 'destructive',
        });
        setUploadResults(acceptedFiles.map((f) => ({ name: f.name, status: 'error' as const, message: error?.message ?? 'Network error' })));
      } finally {
        setUploading(false);
      }
    },
    [onUploadComplete]
  );

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

  const getFileIcon = (fileName?: string) => {
    if (!fileName) return <FileText className="w-4 h-4" />;
    const ext = fileName.trim().split('.').pop()?.toLowerCase() ?? '';
    if (['pdf', 'docx', 'pptx', 'txt'].includes(ext)) return <FileText className="w-4 h-4" />;
    if (['jpg', 'jpeg', 'png'].includes(ext)) return <Image className="w-4 h-4" />;
    return <FileText className="w-4 h-4" />;
  };

  return (
    <div className="space-y-6">
      <Card className="glass border-2 border-dashed border-primary/30 hover:border-primary/60 transition-all duration-300">
        <div
          {...getRootProps()}
          className={`p-8 text-center cursor-pointer transition-all duration-300 ${isDragActive ? 'bg-primary/10 scale-105' : 'hover:bg-primary/5'}`}
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
                <h3 className="text-xl font-semibold gradient-text">{isDragActive ? 'Drop files here' : 'Upload Documents'}</h3>
                <p className="text-muted-foreground max-w-md">Drag & drop your documents here, or click to browse. Supports PDF, DOCX, PPTX, TXT, JPG, PNG files.</p>
                <Button variant="hero" size="lg" className="mt-4">Browse Files</Button>
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
                <div key={index} className="flex items-center space-x-3 p-3 rounded-lg bg-muted/30">
                  {getFileIcon(file?.name)}
                  <span className="flex-1 font-medium">{file?.name ?? 'Unnamed file'}</span>
                  {file.status === 'success' ? <CheckCircle className="w-5 h-5 text-success" /> : <XCircle className="w-5 h-5 text-destructive" />}
                  <span className={`text-sm ${file.status === 'success' ? 'text-success' : 'text-destructive'}`}>{file.message}</span>
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
