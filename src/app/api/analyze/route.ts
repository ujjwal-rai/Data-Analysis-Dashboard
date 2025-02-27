import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: Request) {
  try {
    const { filepath } = await request.json();
    
    // Run Python analysis script
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'src/python/analyze.py'),
      filepath
    ]);

    let result = '';
    let error = '';

    // Collect data from Python script
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    // Wait for Python script to complete
    await new Promise((resolve, reject) => {
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve(result);
        } else {
          reject(new Error(`Python process exited with code ${code}`));
        }
      });
    });

    const analysisResults = JSON.parse(result);

    return NextResponse.json(analysisResults);
  } catch (error) {
    console.error('Analysis error:', error);
    return NextResponse.json(
      { error: 'Failed to analyze data' },
      { status: 500 }
    );
  }
} 