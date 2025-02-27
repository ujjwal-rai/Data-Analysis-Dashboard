'use client';

import React, { useState } from 'react';
import { Upload, Button, Input, message, Spin, ConfigProvider, theme } from 'antd';
import { UploadOutlined, SendOutlined, RobotOutlined, DatabaseOutlined, CloseOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';

export default function ChatInterface() {
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<Array<{role: string, content: string}>>([]);
  const [question, setQuestion] = useState('');
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);

  // Upload props configuration
  const props: UploadProps = {
    name: 'file',
    action: '/api/upload',
    onChange(info) {
      if (info.file.status === 'done') {
        message.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
  };

  // Chat handling
  const handleAskQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const response = await fetch('/api/ask-chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      const data = await response.json();
      setChatHistory([...chatHistory, 
        { role: 'user', content: question },
        { role: 'assistant', content: data.reply }
      ]);
      setQuestion('');
    } catch (error) {
      message.error('Failed to get response from chatbot');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ConfigProvider theme={{ algorithm: theme.darkAlgorithm }}>
      {/* Main Upload Section */}
      <div className="max-w-7xl mx-auto p-4">
        <div className="bg-slate-800 rounded-lg p-6 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <DatabaseOutlined className="text-xl text-blue-500" />
            <h2 className="text-xl font-semibold">Dataset Upload</h2>
          </div>
          <Upload {...props} accept=".csv">
            <Button icon={<UploadOutlined />}>Upload Dataset (CSV)</Button>
          </Upload>
        </div>
      </div>

      {/* Fixed Chat Button */}
      <button
        onClick={() => setIsAssistantOpen(true)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-blue-600 text-white flex items-center justify-center shadow-lg hover:bg-blue-700 transition-all"
        style={{ zIndex: 1000 }}
      >
        <RobotOutlined className="text-xl" />
      </button>

      {/* Chat Popup */}
      {isAssistantOpen && (
        <div className="fixed bottom-24 right-6 w-80 bg-white rounded-lg shadow-xl" style={{ zIndex: 1000 }}>
          {/* Header */}
          <div className="flex justify-between items-center p-4 border-b">
            <span className="font-semibold">AI Assistant</span>
            <button onClick={() => setIsAssistantOpen(false)} className="text-gray-500 hover:text-gray-700">
              <CloseOutlined />
            </button>
          </div>

          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-4 bg-gray-50">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`mb-2 p-2 rounded ${
                msg.role === 'user' ? 'bg-blue-500 text-white ml-auto' : 'bg-gray-200'
              } max-w-[80%] ${msg.role === 'user' ? 'ml-auto' : ''}`}>
                {msg.content}
              </div>
            ))}
          </div>

          {/* Input Area */}
          <div className="p-4 border-t">
            <div className="flex gap-2">
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onPressEnter={handleAskQuestion}
                placeholder="Type your question..."
              />
              <Button 
                onClick={handleAskQuestion} 
                type="primary"
                icon={<SendOutlined />}
                loading={loading}
              />
            </div>
          </div>
        </div>
      )}
    </ConfigProvider>
  );
} 