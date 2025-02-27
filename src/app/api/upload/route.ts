import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { writeFile } from 'fs/promises';
import path from 'path';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file received' },
        { status: 400 }
      );
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Save file to uploads directory
    const uploadDir = path.join(process.cwd(), 'uploads');
    const filepath = path.join(uploadDir, file.name);
    await writeFile(filepath, buffer);

    return NextResponse.json({ 
      message: 'File uploaded successfully',
      filepath 
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Error uploading file' },
      { status: 500 }
    );
  }
} 