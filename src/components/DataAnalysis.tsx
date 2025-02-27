'use client';

import React, { useState } from 'react';
import { Upload, Button, Card, Table, Spin, message, Input, Modal } from 'antd';
import { 
  UploadOutlined, 
  BarChartOutlined, 
  RobotOutlined,
  SendOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons';
import dynamic from 'next/dynamic';

// Define types for our data structures
interface Visualization {
  title: string;
  data: any[];
  layout?: any;
}

interface CorrelationMatrix {
  columns: string[];
  values: number[][];
}

interface DatasetInfo {
  columns: string[];
  shape: [number, number];
  dtypes: { [key: string]: string };
  summary: any;
  sample: any;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

// First, let's add proper typing for the numerical summary
interface NumericalSummary {
  [key: string]: {
    [column: string]: number;
  };
}

const Plot = dynamic(
  () => import('react-plotly.js').then((Plotly) => Plotly.default),
  { ssr: false, loading: () => <div>Loading Plot...</div> }
);

export default function DataAnalysis() {
  const [loading, setLoading] = useState(false);
  const [dataTypeSummary, setDataTypeSummary] = useState<any[]>([]);
  const [numericalSummary, setNumericalSummary] = useState<NumericalSummary | null>(null);
  const [visualizations, setVisualizations] = useState<Visualization[]>([]);
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationMatrix | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

  const handleUpload = async (info: any) => {
    if (info.file.status === 'uploading') {
      setLoading(true);
      return;
    }
    if (info.file.status === 'done') {
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          body: JSON.stringify({ filepath: info.file.response.filepath }),
          headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.error) {
          throw new Error(data.error);
        }
        
        setDataTypeSummary(data.datatypes || []);
        setNumericalSummary(data.numerical_summary);
        setVisualizations(data.visualizations || []);
        setCorrelationMatrix(data.correlation);
        setDatasetInfo(data.dataset_info);
        message.success('Analysis completed successfully');
      } catch (error) {
        message.error('Failed to analyze data');
        console.error('Analysis error:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim() || !datasetInfo) return;
    
    try {
      const response = await fetch('/api/ask-chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question,
          datasetInfo 
        })
      });
      
      const data = await response.json();
      setChatHistory([...chatHistory, 
        { role: 'user', content: question },
        { role: 'assistant', content: data.reply }
      ]);
      setQuestion('');
    } catch (error) {
      message.error('Failed to get response');
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-4 space-y-8">
      {/* Header Card */}
      <Card className="bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500 shadow-2xl hover:shadow-3xl transition-all duration-300">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-4 text-white">
              Data Analysis Dashboard
            </h1>
            <p className="text-white/80 text-lg">
              Upload your dataset and get instant insights
            </p>
          </div>
          <Upload
            action="/api/upload"
            onChange={handleUpload}
            accept=".csv"
            maxCount={1}
          >
            <Button 
              icon={<UploadOutlined />} 
              size="large"
              className="bg-white hover:bg-white/90 text-purple-600 border-none hover:scale-105 transition-all duration-200"
            >
              Upload Dataset (CSV)
            </Button>
          </Upload>
        </div>
      </Card>

      {/* Loading State */}
      {loading && (
        <div className="flex justify-center p-8">
          <Spin size="large" />
        </div>
      )}

      {/* Analysis Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {dataTypeSummary.length > 0 && (
          <Card 
            title={<span className="text-lg font-bold text-violet-600">Data Types</span>}
            className="shadow-xl hover:shadow-2xl transition-all duration-300 border-t-4 border-violet-500"
          >
            <Table 
              dataSource={dataTypeSummary}
              columns={[
                { title: 'Column', dataIndex: 'Column' },
                { title: 'Type', dataIndex: 'Pandas Dtype' },
                { title: 'Category', dataIndex: 'Custom Category' }
              ]}
              pagination={false}
            />
          </Card>
        )}

        {numericalSummary && (
          <Card 
            title={<span className="text-lg font-bold text-fuchsia-600">Statistics</span>}
            className="shadow-xl hover:shadow-2xl transition-all duration-300 border-t-4 border-fuchsia-500"
          >
            <Table 
              dataSource={Object.entries(numericalSummary).map(([key, value]) => ({
                metric: key,
                ...(value as { [key: string]: number })  // Type assertion here
              }))}
              columns={[
                { title: 'Metric', dataIndex: 'metric' },
                ...Object.keys(numericalSummary[Object.keys(numericalSummary)[0]]).map(col => ({
                  title: col,
                  dataIndex: col
                }))
              ]}
              pagination={false}
            />
          </Card>
        )}
      </div>

      {/* Visualizations */}
      {visualizations.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {visualizations.map((viz, index) => (
            <Card 
              key={index} 
              title={<span className="text-lg font-bold text-purple-600">{viz.title}</span>}
              className="shadow-xl hover:shadow-2xl transition-all duration-300 border-t-4 border-purple-500"
            >
              <Plot
                data={viz.data}
                layout={{
                  ...viz.layout,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { color: '#4a5568' }
                }}
                config={{ responsive: true }}
              />
            </Card>
          ))}

          {correlationMatrix && (
            <Card 
              title={<span className="text-lg font-bold text-pink-600">Correlation Heatmap</span>}
              className="shadow-xl hover:shadow-2xl transition-all duration-300 border-t-4 border-pink-500"
            >
              <Plot
                data={[{
                  type: 'heatmap',
                  z: correlationMatrix.values,
                  x: correlationMatrix.columns,
                  y: correlationMatrix.columns,
                  colorscale: 'Viridis'
                }]}
                layout={{
                  height: 500,
                  width: '100%',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { color: '#4a5568' }
                }}
              />
            </Card>
          )}
        </div>
      )}

      {/* Chat Button */}
      {datasetInfo && (
        <>
          <div className="fixed inset-x-0 bottom-0 left-0 p-8 pointer-events-none z-[9999]">
            <button
              onClick={() => setIsChatOpen(true)}
              className="w-16 h-16 rounded-full bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white flex items-center justify-center shadow-xl hover:shadow-2xl transition-all duration-200 hover:scale-110 pointer-events-auto"
              style={{
                boxShadow: '0 0 20px rgba(123, 37, 235, 0.3)',
              }}
            >
              <RobotOutlined className="text-2xl" />
            </button>
          </div>

          {/* Chat Modal */}
          <Modal
            title={
              <div className="flex items-center gap-2">
                <RobotOutlined className="text-purple-500 text-2xl" />
                <span className="text-xl">Dataset Assistant</span>
              </div>
            }
            open={isChatOpen}
            onCancel={() => setIsChatOpen(false)}
            footer={null}
            width={600}
            className="rounded-xl"
            style={{ 
              position: 'fixed',
              bottom: '120px',
              left: '32px',
              top: 'auto',
              transform: 'none',
              zIndex: 9998
            }}
          >
            <div className="h-96 overflow-y-auto mb-4 p-4 bg-gray-50 rounded-lg">
              {chatHistory.map((msg, idx) => (
                <div
                  key={idx}
                  className={`mb-3 p-3 rounded-lg ${
                    msg.role === 'user'
                      ? 'bg-purple-500 text-white ml-auto'
                      : 'bg-gray-200'
                  } max-w-[80%] ${msg.role === 'user' ? 'ml-auto' : ''}`}
                >
                  {msg.content}
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onPressEnter={handleAskQuestion}
                placeholder="Ask about your dataset..."
                prefix={<QuestionCircleOutlined className="text-gray-400" />}
              />
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleAskQuestion}
                className="bg-purple-500 hover:bg-purple-600"
              />
            </div>
          </Modal>
        </>
      )}
    </div>
  );
} 