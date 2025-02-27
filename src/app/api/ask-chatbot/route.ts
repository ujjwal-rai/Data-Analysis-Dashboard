import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { question, datasetInfo } = await request.json();

    // Create a context-aware prompt
    const contextPrompt = `
You are a data analysis assistant. Here's information about the dataset:
- Columns: ${datasetInfo.columns.join(', ')}
- Shape: ${datasetInfo.shape[0]} rows, ${datasetInfo.shape[1]} columns
- Data types: ${JSON.stringify(datasetInfo.dtypes)}
- Sample data: ${JSON.stringify(datasetInfo.sample)}

Please analyze this information and answer the following question:
${question}

Provide a clear, concise answer based on the dataset information provided.`;

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: "llama3-8b-8192",
        messages: [{ role: "user", content: contextPrompt }],
        max_tokens: 500,
        temperature: 0.7
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error?.message || 'API request failed');
    }

    return NextResponse.json({ 
      reply: data.choices[0].message.content 
    });
  } catch (error) {
    console.error('Chatbot API error:', error);
    return NextResponse.json(
      { error: 'Failed to get response from chatbot' },
      { status: 500 }
    );
  }
} 