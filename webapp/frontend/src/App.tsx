import { useState, useCallback } from 'react'
import { Upload, FileText, Loader2, BarChart3, Settings2 } from 'lucide-react'
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const API_URL = 'http://localhost:8000'

interface TextChunk {
  text: string
  index: number
}

interface EmbeddingPoint {
  x: number
  y: number
  z?: number
  text: string
  index: number
}

interface PDFUploadResponse {
  text_chunks: string[]
  total_pages: number
  total_characters: number
}

interface EmbeddingResponse {
  reduced_embeddings: number[][]
  original_dim: number
  n_chunks: number
  text_chunks: string[]
  reduction_method: string
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [textChunks, setTextChunks] = useState<TextChunk[]>([])
  const [embeddings, setEmbeddings] = useState<EmbeddingPoint[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [step, setStep] = useState<'upload' | 'configure' | 'visualize'>('upload')
  const [reductionMethod, setReductionMethod] = useState<'pca' | 'tsne'>('pca')
  const [chunkSize, setChunkSize] = useState(500)
  const [selectedPoint, setSelectedPoint] = useState<EmbeddingPoint | null>(null)
  const [pdfInfo, setPdfInfo] = useState<{ pages: number; chars: number } | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
      setError(null)
    } else {
      setError('Please select a valid PDF file')
    }
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setFile(droppedFile)
      setError(null)
    } else {
      setError('Please drop a valid PDF file')
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }, [])

  const uploadPDF = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/upload-pdf?chunk_size=${chunkSize}&overlap=50`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to upload PDF')
      }

      const data: PDFUploadResponse = await response.json()
      setTextChunks(data.text_chunks.map((text, index) => ({ text, index })))
      setPdfInfo({ pages: data.total_pages, chars: data.total_characters })
      setStep('configure')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload PDF')
    } finally {
      setLoading(false)
    }
  }

  const generateEmbeddings = async () => {
    if (textChunks.length === 0) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/generate-embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text_chunks: textChunks.map(c => c.text),
          reduction_method: reductionMethod,
          n_components: 2,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to generate embeddings')
      }

      const data: EmbeddingResponse = await response.json()
      const points: EmbeddingPoint[] = data.reduced_embeddings.map((coords, index) => ({
        x: coords[0],
        y: coords[1],
        z: coords[2],
        text: data.text_chunks[index],
        index,
      }))
      setEmbeddings(points)
      setStep('visualize')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate embeddings')
    } finally {
      setLoading(false)
    }
  }

  const resetApp = () => {
    setFile(null)
    setTextChunks([])
    setEmbeddings([])
    setSelectedPoint(null)
    setPdfInfo(null)
    setStep('upload')
    setError(null)
  }

  const colors = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#6366f1'
  ]

  const getColor = (index: number) => colors[index % colors.length]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-2">
            DeepSeek-V3 Embedding Visualizer
          </h1>
          <p className="text-slate-400">
            Upload a PDF and visualize text embeddings using dimensionality reduction
          </p>
        </header>

        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            <StepIndicator 
              number={1} 
              label="Upload" 
              active={step === 'upload'} 
              completed={step !== 'upload'} 
            />
            <div className="w-12 h-0.5 bg-slate-600" />
            <StepIndicator 
              number={2} 
              label="Configure" 
              active={step === 'configure'} 
              completed={step === 'visualize'} 
            />
            <div className="w-12 h-0.5 bg-slate-600" />
            <StepIndicator 
              number={3} 
              label="Visualize" 
              active={step === 'visualize'} 
              completed={false} 
            />
          </div>
        </div>

        {error && (
          <div className="max-w-2xl mx-auto mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
            {error}
          </div>
        )}

        {step === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="border-2 border-dashed border-slate-600 rounded-xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer bg-slate-800/50"
            >
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
                id="pdf-upload"
              />
              <label htmlFor="pdf-upload" className="cursor-pointer">
                <Upload className="w-16 h-16 mx-auto mb-4 text-slate-400" />
                <p className="text-xl text-white mb-2">
                  {file ? file.name : 'Drop your PDF here'}
                </p>
                <p className="text-slate-400">
                  or click to browse
                </p>
              </label>
            </div>

            {file && (
              <div className="mt-6 p-4 bg-slate-800 rounded-lg flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <FileText className="w-8 h-8 text-blue-400" />
                  <div>
                    <p className="text-white font-medium">{file.name}</p>
                    <p className="text-slate-400 text-sm">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <button
                  onClick={uploadPDF}
                  disabled={loading}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <span>Extract Text</span>
                  )}
                </button>
              </div>
            )}
          </div>
        )}

        {step === 'configure' && (
          <div className="max-w-4xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Settings2 className="w-5 h-5 mr-2" />
                  Configuration
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-slate-300 mb-2">Reduction Method</label>
                    <select
                      value={reductionMethod}
                      onChange={(e) => setReductionMethod(e.target.value as 'pca' | 'tsne')}
                      className="w-full p-3 bg-slate-700 text-white rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="pca">PCA (Principal Component Analysis)</option>
                      <option value="tsne">t-SNE (t-Distributed Stochastic Neighbor Embedding)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-slate-300 mb-2">Chunk Size: {chunkSize} characters</label>
                    <input
                      type="range"
                      min="100"
                      max="1000"
                      step="50"
                      value={chunkSize}
                      onChange={(e) => setChunkSize(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  {pdfInfo && (
                    <div className="p-4 bg-slate-700/50 rounded-lg">
                      <p className="text-slate-300">
                        <span className="font-medium">Pages:</span> {pdfInfo.pages}
                      </p>
                      <p className="text-slate-300">
                        <span className="font-medium">Characters:</span> {pdfInfo.chars.toLocaleString()}
                      </p>
                      <p className="text-slate-300">
                        <span className="font-medium">Chunks:</span> {textChunks.length}
                      </p>
                    </div>
                  )}

                  <button
                    onClick={generateEmbeddings}
                    disabled={loading}
                    className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Generating Embeddings...</span>
                      </>
                    ) : (
                      <>
                        <BarChart3 className="w-4 h-4" />
                        <span>Generate & Visualize</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="bg-slate-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Text Chunks Preview</h3>
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {textChunks.slice(0, 10).map((chunk) => (
                    <div
                      key={chunk.index}
                      className="p-3 bg-slate-700/50 rounded-lg text-sm text-slate-300"
                    >
                      <span className="text-blue-400 font-medium">#{chunk.index + 1}</span>
                      {' '}
                      {chunk.text.substring(0, 100)}...
                    </div>
                  ))}
                  {textChunks.length > 10 && (
                    <p className="text-slate-400 text-center py-2">
                      ... and {textChunks.length - 10} more chunks
                    </p>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={resetApp}
              className="mt-6 px-4 py-2 text-slate-400 hover:text-white transition-colors"
            >
              Start Over
            </button>
          </div>
        )}

        {step === 'visualize' && (
          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-slate-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Embedding Visualization ({reductionMethod.toUpperCase()})
                </h3>
                <div className="h-96 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        name="Component 1" 
                        tick={{ fill: '#94a3b8' }}
                        axisLine={{ stroke: '#475569' }}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        name="Component 2" 
                        tick={{ fill: '#94a3b8' }}
                        axisLine={{ stroke: '#475569' }}
                      />
                      <ZAxis range={[100, 100]} />
                      <Tooltip
                        content={({ payload }) => {
                          if (payload && payload.length > 0) {
                            const data = payload[0].payload as EmbeddingPoint
                            return (
                              <div className="bg-slate-900 p-3 rounded-lg border border-slate-700 max-w-xs">
                                <p className="text-blue-400 font-medium mb-1">Chunk #{data.index + 1}</p>
                                <p className="text-slate-300 text-sm">{data.text.substring(0, 150)}...</p>
                              </div>
                            )
                          }
                          return null
                        }}
                      />
                      <Scatter
                        data={embeddings}
                        onClick={(data) => setSelectedPoint(data as unknown as EmbeddingPoint)}
                      >
                        {embeddings.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={getColor(index)}
                            stroke={selectedPoint?.index === entry.index ? '#fff' : 'transparent'}
                            strokeWidth={2}
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-slate-400 text-sm mt-4">
                  Click on a point to see the full text chunk. Each point represents a text chunk from your PDF,
                  reduced from 2048 dimensions to 2D using {reductionMethod.toUpperCase()}.
                </p>
              </div>

              <div className="bg-slate-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Selected Chunk</h3>
                {selectedPoint ? (
                  <div className="space-y-4">
                    <div className="p-3 bg-slate-700/50 rounded-lg">
                      <p className="text-blue-400 font-medium mb-2">Chunk #{selectedPoint.index + 1}</p>
                      <p className="text-slate-300 text-sm whitespace-pre-wrap">{selectedPoint.text}</p>
                    </div>
                    <div className="text-sm text-slate-400">
                      <p>X: {selectedPoint.x.toFixed(4)}</p>
                      <p>Y: {selectedPoint.y.toFixed(4)}</p>
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-400">Click on a point in the chart to view its content</p>
                )}

                <div className="mt-6 pt-6 border-t border-slate-700">
                  <h4 className="text-white font-medium mb-3">Statistics</h4>
                  <div className="space-y-2 text-sm text-slate-300">
                    <p>Total Chunks: {embeddings.length}</p>
                    <p>Original Dimension: 2048</p>
                    <p>Reduced Dimension: 2</p>
                    <p>Method: {reductionMethod.toUpperCase()}</p>
                  </div>
                </div>

                <div className="mt-6 space-y-2">
                  <button
                    onClick={() => setStep('configure')}
                    className="w-full px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                  >
                    Reconfigure
                  </button>
                  <button
                    onClick={resetApp}
                    className="w-full px-4 py-2 text-slate-400 hover:text-white transition-colors"
                  >
                    Upload New PDF
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        <footer className="mt-16 text-center text-slate-500 text-sm">
          <p>
            Built with DeepSeek-V3 inspired embedding architecture (dim=2048)
          </p>
          <p className="mt-1">
            Reference: inference/model.py - ParallelEmbedding class (lines 87-126)
          </p>
        </footer>
      </div>
    </div>
  )
}

interface StepIndicatorProps {
  number: number
  label: string
  active: boolean
  completed: boolean
}

function StepIndicator({ number, label, active, completed }: StepIndicatorProps) {
  return (
    <div className="flex flex-col items-center">
      <div
        className={`w-10 h-10 rounded-full flex items-center justify-center font-medium transition-colors ${
          active
            ? 'bg-blue-600 text-white'
            : completed
            ? 'bg-green-600 text-white'
            : 'bg-slate-700 text-slate-400'
        }`}
      >
        {number}
      </div>
      <span className={`mt-2 text-sm ${active ? 'text-white' : 'text-slate-400'}`}>
        {label}
      </span>
    </div>
  )
}

export default App
