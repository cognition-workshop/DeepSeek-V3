import { useState, useEffect } from 'react'
import axios from 'axios'
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts'
import { 
  Play, 
  Settings, 
  Loader2, 
  AlertCircle,
  Eye,
  BarChart3,
  Network,
  Zap
} from 'lucide-react'
import './App.css'

const API_BASE_URL = 'http://127.0.0.1:8000'

interface ModelConfig {
  vocab_size: number
  dim: number
  world_size: number
}

interface VocabPartition {
  rank: number
  start_idx: number
  end_idx: number
  size: number
  percentage: number
}

interface EmbeddingData {
  tokens: string[]
  token_ids: number[]
  embeddings: number[][]
  vocab_partitions: {
    partitions: VocabPartition[]
    debug_info: any
  }
  visualization_data: {
    pca: {
      x: number[]
      y: number[]
      z: number[]
      tokens: string[]
      token_ids: number[]
      explained_variance: number[]
    }
    tsne: {
      x: number[]
      y: number[]
      tokens: string[]
      token_ids: number[]
    } | null
    similarity_matrix: number[][]
    embedding_stats: {
      mean: number[]
      std: number[]
      norm: number[]
    }
  }
}

function App() {
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [inputText, setInputText] = useState('Hello world, this is a test of DeepSeek embeddings!')
  const [worldSize, setWorldSize] = useState(1)
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null)
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null)
  const [activeTab, setActiveTab] = useState<'pca' | 'tsne' | 'similarity' | 'partitions'>('pca')

  useEffect(() => {
    checkModelStatus()
  }, [])

  const checkModelStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`)
      setIsModelLoaded(response.data.model_loaded)
    } catch (err) {
      setError('Failed to connect to backend API')
    }
  }

  const initializeModel = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_BASE_URL}/initialize`, { world_size: worldSize })
      setModelConfig(response.data.config)
      setIsModelLoaded(true)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to initialize model')
    } finally {
      setIsLoading(false)
    }
  }

  const processText = async () => {
    if (!inputText.trim()) return
    
    setIsLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_BASE_URL}/embed`, {
        text: inputText,
        world_size: worldSize
      })
      setEmbeddingData(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to process text')
    } finally {
      setIsLoading(false)
    }
  }

  const renderPCAVisualization = () => {
    if (!embeddingData?.visualization_data.pca) return null

    const { pca } = embeddingData.visualization_data
    const data = pca.x.map((x, i) => ({
      x,
      y: pca.y[i],
      token: pca.tokens[i],
      token_id: pca.token_ids[i]
    }))

    return (
      <div className="visualization-panel">
        <h3 className="text-lg font-semibold mb-4">PCA Visualization</h3>
        <div className="mb-4">
          <p className="text-sm text-gray-600">
            Explained Variance: {pca.explained_variance.map(v => (v * 100).toFixed(1)).join('%, ')}%
          </p>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} />
            <YAxis dataKey="y" type="number" domain={['dataMin', 'dataMax']} />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload[0]) {
                  const data = payload[0].payload
                  return (
                    <div className="bg-white p-2 border rounded shadow">
                      <p><strong>Token:</strong> {data.token}</p>
                      <p><strong>ID:</strong> {data.token_id}</p>
                      <p><strong>Position:</strong> ({data.x.toFixed(3)}, {data.y.toFixed(3)})</p>
                    </div>
                  )
                }
                return null
              }}
            />
            <Scatter dataKey="y" fill="#3b82f6" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    )
  }

  const renderTSNEVisualization = () => {
    if (!embeddingData?.visualization_data.tsne) {
      return (
        <div className="visualization-panel">
          <h3 className="text-lg font-semibold mb-4">t-SNE Visualization</h3>
          <p className="text-gray-500">t-SNE requires at least 4 tokens. Add more text to enable this visualization.</p>
        </div>
      )
    }

    const { tsne } = embeddingData.visualization_data
    const data = tsne.x.map((x, i) => ({
      x,
      y: tsne.y[i],
      token: tsne.tokens[i],
      token_id: tsne.token_ids[i]
    }))

    return (
      <div className="visualization-panel">
        <h3 className="text-lg font-semibold mb-4">t-SNE Visualization</h3>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} />
            <YAxis dataKey="y" type="number" domain={['dataMin', 'dataMax']} />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload[0]) {
                  const data = payload[0].payload
                  return (
                    <div className="bg-white p-2 border rounded shadow">
                      <p><strong>Token:</strong> {data.token}</p>
                      <p><strong>ID:</strong> {data.token_id}</p>
                      <p><strong>Position:</strong> ({data.x.toFixed(3)}, {data.y.toFixed(3)})</p>
                    </div>
                  )
                }
                return null
              }}
            />
            <Scatter dataKey="y" fill="#10b981" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    )
  }

  const renderSimilarityMatrix = () => {
    if (!embeddingData?.visualization_data.similarity_matrix) return null

    const { similarity_matrix } = embeddingData.visualization_data
    const { tokens } = embeddingData

    return (
      <div className="visualization-panel">
        <h3 className="text-lg font-semibold mb-4">Token Similarity Matrix</h3>
        <div className="overflow-auto max-h-96">
          <table className="min-w-full text-xs">
            <thead>
              <tr>
                <th className="p-1"></th>
                {tokens.map((token, i) => (
                  <th key={i} className="p-1 text-center min-w-16">{token}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {similarity_matrix.map((row, i) => (
                <tr key={i}>
                  <td className="p-1 font-medium">{tokens[i]}</td>
                  {row.map((similarity, j) => (
                    <td 
                      key={j} 
                      className="p-1 text-center"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${Math.abs(similarity)})`
                      }}
                    >
                      {similarity.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const renderVocabPartitions = () => {
    if (!embeddingData?.vocab_partitions.partitions) return null

    const { partitions } = embeddingData.vocab_partitions
    const data = partitions.map(p => ({
      rank: `Rank ${p.rank}`,
      size: p.size,
      percentage: p.percentage,
      range: `${p.start_idx}-${p.end_idx}`
    }))

    return (
      <div className="visualization-panel">
        <h3 className="text-lg font-semibold mb-4">Vocabulary Partitioning</h3>
        <div className="mb-4">
          <p className="text-sm text-gray-600">
            Total Vocabulary Size: {modelConfig?.vocab_size.toLocaleString()}
          </p>
          <p className="text-sm text-gray-600">
            World Size: {worldSize} processes
          </p>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="rank" />
            <YAxis />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload[0]) {
                  const data = payload[0].payload
                  return (
                    <div className="bg-white p-2 border rounded shadow">
                      <p><strong>{data.rank}</strong></p>
                      <p><strong>Size:</strong> {data.size.toLocaleString()}</p>
                      <p><strong>Percentage:</strong> {data.percentage.toFixed(1)}%</p>
                      <p><strong>Range:</strong> {data.range}</p>
                    </div>
                  )
                }
                return null
              }}
            />
            <Bar dataKey="size" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            DeepSeek-V3 Embedding Visualizer
          </h1>
          <p className="text-gray-600">
            Explore how the DeepSeek-V3 model processes text through its ParallelEmbedding layer
          </p>
        </header>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Text Input & Configuration</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Input Text
                </label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={3}
                  placeholder="Enter text to analyze..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  World Size (Distributed Processing)
                </label>
                <select
                  value={worldSize}
                  onChange={(e) => setWorldSize(Number(e.target.value))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={1}>1 (Single Process)</option>
                  <option value={2}>2 Processes</option>
                  <option value={4}>4 Processes</option>
                  <option value={8}>8 Processes</option>
                </select>
              </div>

              <div className="flex space-x-4">
                {!isModelLoaded ? (
                  <button
                    onClick={initializeModel}
                    disabled={isLoading}
                    className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Settings className="h-4 w-4 mr-2" />
                    )}
                    Initialize Model
                  </button>
                ) : (
                  <button
                    onClick={processText}
                    disabled={isLoading || !inputText.trim()}
                    className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Process Text
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Model Status</h2>
            
            <div className="space-y-3">
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full mr-3 ${isModelLoaded ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm">
                  Model: {isModelLoaded ? 'Loaded' : 'Not Loaded'}
                </span>
              </div>

              {modelConfig && (
                <>
                  <div className="text-sm text-gray-600">
                    <strong>Vocabulary Size:</strong> {modelConfig.vocab_size.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-600">
                    <strong>Embedding Dimension:</strong> {modelConfig.dim}
                  </div>
                  <div className="text-sm text-gray-600">
                    <strong>World Size:</strong> {modelConfig.world_size}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {embeddingData && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold">Embedding Visualizations</h2>
              
              <div className="flex space-x-2">
                <button
                  onClick={() => setActiveTab('pca')}
                  className={`flex items-center px-3 py-2 rounded-lg text-sm ${
                    activeTab === 'pca' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <BarChart3 className="h-4 w-4 mr-1" />
                  PCA
                </button>
                <button
                  onClick={() => setActiveTab('tsne')}
                  className={`flex items-center px-3 py-2 rounded-lg text-sm ${
                    activeTab === 'tsne' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Eye className="h-4 w-4 mr-1" />
                  t-SNE
                </button>
                <button
                  onClick={() => setActiveTab('similarity')}
                  className={`flex items-center px-3 py-2 rounded-lg text-sm ${
                    activeTab === 'similarity' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Network className="h-4 w-4 mr-1" />
                  Similarity
                </button>
                <button
                  onClick={() => setActiveTab('partitions')}
                  className={`flex items-center px-3 py-2 rounded-lg text-sm ${
                    activeTab === 'partitions' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Zap className="h-4 w-4 mr-1" />
                  Partitions
                </button>
              </div>
            </div>

            {activeTab === 'pca' && renderPCAVisualization()}
            {activeTab === 'tsne' && renderTSNEVisualization()}
            {activeTab === 'similarity' && renderSimilarityMatrix()}
            {activeTab === 'partitions' && renderVocabPartitions()}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
