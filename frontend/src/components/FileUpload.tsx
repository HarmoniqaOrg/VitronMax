import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setJobId(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setJobId(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE;
      if (!apiBaseUrl) {
        throw new Error('API base URL is not configured. Please set VITE_API_BASE in your .env file.');
      }
      const response = await fetch(`${apiBaseUrl}/batch_predict_csv`, {
        method: 'POST',
        body: formData,
      });

      const responseText = await response.text(); // Read response as text first
      console.log('Raw server response:', responseText); // Log it

      if (!response.ok) {
        // Try to parse error as JSON, but fall back to raw text if it fails
        let errorData;
        try {
          errorData = JSON.parse(responseText);
        } catch (e) {
          // If parsing fails, use the raw text as the error message
          throw new Error(`Server error: ${response.status} ${response.statusText}. Response: ${responseText}`);
        }
        throw new Error(errorData.detail || `Server error: ${response.status} ${response.statusText}`);
      }

      // Now try to parse the (logged) text as JSON
      const result = JSON.parse(responseText);
      setJobId(result.job_id);
      // setSelectedFile(null); // Optional: Clear file input after successful upload

    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Upload CSV for Batch Prediction</CardTitle>
        <CardDescription>Select a CSV file containing SMILES strings for BBB permeability prediction.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <Label htmlFor="csv-file">CSV File</Label>
          <Input id="csv-file" type="file" accept=".csv" onChange={handleFileChange} />
        </div>
        {selectedFile && (
          <p className="text-sm text-muted-foreground">
            Selected file: {selectedFile.name}
          </p>
        )}
      </CardContent>
      <CardFooter className="flex flex-col items-start space-y-2">
        <Button onClick={handleSubmit} disabled={!selectedFile || isLoading}>
          {isLoading ? 'Uploading...' : 'Upload and Predict'}
        </Button>
        {jobId && (
          <div className="p-2 bg-green-100 text-green-700 border border-green-300 rounded-md text-sm">
            <p>Batch job started successfully!</p>
            <p>Job ID: <strong>{jobId}</strong></p>
            <p>You can track its status on the 'Jobs List' page.</p>
          </div>
        )}
        {error && (
          <div className="p-2 bg-red-100 text-red-700 border border-red-300 rounded-md text-sm">
            <p>Error: {error}</p>
          </div>
        )}
      </CardFooter>
    </Card>
  );
};

export default FileUpload;
