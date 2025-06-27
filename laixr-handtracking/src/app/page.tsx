"use client";

import { useState, useCallback, useEffect, useRef, ElementType, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Video,
  BarChart3,
  FileText,
  Settings,
  CheckCircle,
  Clock,
  FileVideo,
  ChevronRight,
  BrainCircuit,
  Database,
  Gauge,
  Search,
  X,
  Loader2,
  AlertCircle,
  Eye,
  RefreshCw,
  XCircle,
  Download,
  Activity,
  ThumbsUp,
  ListChecks,
  Sparkles,
  ArrowRightLeft,
  Play,
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Slider } from '@/components/ui/slider';
import AnalysisReport from '@/components/ui/AnalysisReport';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';


// Backend configuration
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

interface AnalysisParameters {
  min_detection_confidence: number;
  min_tracking_confidence: number;
  filter_min_cutoff: number;
  filter_beta: number;
  // Outlier detection settings
  outlier_detection_enabled: boolean;
  outlier_detection_method: 'iqr' | 'zscore';
  outlier_threshold: number;
  // Advanced filtering settings  
  fingertip_filter_multiplier: number;
  joint_filter_multiplier: number;
  palm_responsiveness_multiplier: number;
}

interface AnalysisResult {
  dexterity_score?: number;
  kinematics?: any;
  [key: string]: any;
}

interface BackendAnalysis {
  id: string;
  original_name: string;
  status: 'processing' | 'completed' | 'failed';
  created_at: string;
  error_message?: string;
  results?: string; // Stored as a JSON string
  analysis_parameters?: string; // Stored as a JSON string
}

interface ProcessedFile {
  id: string;
  name: string;
  date: string;
  status: 'Completed' | 'Processing' | 'Failed' | 'Uploading';
  analysisResult?: AnalysisResult;
  analysisParameters?: AnalysisParameters;
  errorMessage?: string;
}

interface Stats {
  totalAnalyses: number;
  processing: number;
  completed: number;
  failed: number;
  avgDexterity: number;
  systemStatus: 'Online' | 'Offline';
  statusDistribution: { name: string; value: number; color: string }[];
  weeklyActivity: { date: string; count: number }[];
}

const getStatusChip = (file: ProcessedFile) => {
    const statusConfig: { [key in ProcessedFile['status']]: { icon: ElementType, color: string, animate?: boolean } } = {
      Completed: { icon: CheckCircle, color: 'text-green-400' },
      Processing: { icon: Loader2, color: 'text-blue-400', animate: true },
      Uploading: { icon: Loader2, color: 'text-yellow-400', animate: true },
      Failed: { icon: XCircle, color: 'text-red-400' },
    };
    const config = statusConfig[file.status];
    if (!config) return null;
    const Icon = config.icon;
    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger>
                    <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-secondary/40 ${config.color}`}>
                        <Icon className={`w-3.5 h-3.5 ${config.animate ? 'animate-spin' : ''}`} />
                        {file.status}
                    </div>
                </TooltipTrigger>
                {file.status === 'Failed' && file.errorMessage && (
                    <TooltipContent className="bg-destructive text-destructive-foreground max-w-xs">
                        <p>{file.errorMessage}</p>
                    </TooltipContent>
                )}
            </Tooltip>
        </TooltipProvider>
    );
};

const LiveAnalysisViewer = ({ analysisId, onClose }: { analysisId: string; onClose: () => void; }) => {
    const [frame, setFrame] = useState<string | null>(null);
    const [status, setStatus] = useState<'connecting' | 'streaming' | 'completed' | 'error'>('connecting');
    const ws = useRef<WebSocket | null>(null);
    const statusRef = useRef(status);

    useEffect(() => {
        statusRef.current = status;
    }, [status]);

    useEffect(() => {
        const socket = new WebSocket(`${BACKEND_URL.replace(/^http/, 'ws')}/ws/live/${analysisId}`);
        ws.current = socket;

        socket.onopen = () => setStatus('streaming');
        socket.onerror = () => setStatus('error');

        socket.onmessage = (event) => {
            const newFrameUrl = URL.createObjectURL(event.data);
            setFrame(oldFrameUrl => {
              if (oldFrameUrl) URL.revokeObjectURL(oldFrameUrl);
              return newFrameUrl;
            });
        };

        socket.onclose = () => {
            // The live feed is over. Set a neutral 'completed' status.
            // The main page polling will fetch the final 'failed' or 'completed' status.
            setStatus('completed');
        };

        return () => {
            // Cleanup: close the socket when the component unmounts.
            socket.close();
        };
    }, [analysisId]);

    const handleClose = () => {
        if (ws.current?.readyState === WebSocket.OPEN) {
             ws.current.close();
        }
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50">
            <div className="bg-card rounded-lg max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl">
                <div className="p-4 border-b flex justify-between items-center">
                    <h3 className="font-semibold flex items-center gap-2">
                        <Video className="w-5 h-5 text-primary" />
                        Live Analysis Feed: <span className="font-mono text-sm">{analysisId.slice(0, 8)}...</span>
                    </h3>
                    <button onClick={handleClose} className="p-1 rounded-md hover:bg-primary/10"><X className="w-5 h-5" /></button>
                </div>
                <div className="p-6 flex-grow flex items-center justify-center bg-background/50">
                    {status === 'connecting' && <div className="flex flex-col items-center gap-2"><Loader2 className="w-8 h-8 animate-spin" /><span>Connecting to stream...</span></div>}
                    {status === 'error' && <div className="flex flex-col items-center gap-2 text-destructive"><AlertCircle className="w-10 h-10" /><span className="mt-2 font-semibold">Connection Error</span><p className="text-sm">Could not establish live feed.</p><button onClick={handleClose} className="mt-4 px-4 py-2 bg-primary/90 text-primary-foreground rounded-md hover:bg-primary">Close</button></div>}
                    {status === 'completed' && <div className="flex flex-col items-center gap-2 text-green-600"><CheckCircle className="w-10 h-10" /><span className="mt-2 font-semibold">Analysis Completed</span><p className="text-sm">The live feed has ended.</p><button onClick={handleClose} className="mt-4 px-4 py-2 bg-primary/90 text-primary-foreground rounded-md hover:bg-primary">Close & View Report</button></div>}
                    {status === 'streaming' && !frame && <div className="flex flex-col items-center gap-2"><Loader2 className="w-8 h-8 animate-spin" /><span>Waiting for first frame...</span></div>}
                    {status === 'streaming' && frame && (
                        <img src={frame} alt="Live analysis feed" className="max-w-full max-h-full object-contain rounded-md" />
                    )}
                </div>
            </div>
        </div>
    );
};

const VideoModal = ({ analysisId, fileName, onClose }: { analysisId: string; fileName: string; onClose: () => void; }) => {
    const [error, setError] = useState<string | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    const videoUrl = `${BACKEND_URL}/api/analysis/${analysisId}/video`;

    const handleVideoError = (event: any) => {
        console.error('Video error event:', event);
        const video = event.target;
        const errorDetails = {
            error: video.error ? {
                code: video.error.code,
                message: video.error.message,
                MEDIA_ERR_ABORTED: video.error.MEDIA_ERR_ABORTED,
                MEDIA_ERR_NETWORK: video.error.MEDIA_ERR_NETWORK,
                MEDIA_ERR_DECODE: video.error.MEDIA_ERR_DECODE,
                MEDIA_ERR_SRC_NOT_SUPPORTED: video.error.MEDIA_ERR_SRC_NOT_SUPPORTED
            } : null,
            networkState: video.networkState,
            readyState: video.readyState,
            src: video.src,
            currentSrc: video.currentSrc
        };
        console.error('Video error details:', errorDetails);
        
        let errorMessage = 'Failed to load video. ';
        if (video.error) {
            switch (video.error.code) {
                case 1:
                    errorMessage += 'Video loading was aborted.';
                    break;
                case 2:
                    errorMessage += 'Network error occurred while loading video.';
                    break;
                case 3:
                    errorMessage += 'Video format is not supported or corrupted.';
                    break;
                case 4:
                    errorMessage += 'Video source is not supported by your browser.';
                    break;
                default:
                    errorMessage += 'Unknown error occurred.';
            }
        } else {
            errorMessage += 'The video file may be corrupted or incompatible.';
        }
        
        setError(errorMessage);
    };

    const handleVideoLoad = () => {
        setError(null);
    };

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50">
            <div className="bg-card rounded-lg max-w-6xl w-full max-h-[90vh] flex flex-col shadow-2xl">
                <div className="p-4 border-b flex justify-between items-center">
                    <h3 className="font-semibold flex items-center gap-2">
                        <Play className="w-5 h-5 text-primary" />
                        Video with Hand Tracking: <span className="font-mono text-sm">{fileName}</span>
                    </h3>
                    <button onClick={onClose} className="p-1 rounded-md hover:bg-primary/10">
                        <X className="w-5 h-5" />
                    </button>
                </div>
                <div className="p-6 flex-grow flex items-center justify-center bg-background/50">
                    {error ? (
                        <div className="flex flex-col items-center gap-4 text-destructive">
                            <AlertCircle className="w-16 h-16" />
                            <div className="text-center">
                                <p className="font-semibold text-lg">Video Playback Error</p>
                                <p className="text-sm mt-2">{error}</p>
                            </div>
                            <button 
                                onClick={onClose} 
                                className="mt-4 px-4 py-2 bg-primary/90 text-primary-foreground rounded-md hover:bg-primary"
                            >
                                Close
                            </button>
                        </div>
                    ) : (
                        <video
                            ref={videoRef}
                            src={videoUrl}
                            className="max-w-full max-h-full rounded-lg"
                            controls
                            autoPlay
                            onError={handleVideoError}
                            onLoadedData={handleVideoLoad}
                            playsInline
                        >
                            Your browser does not support the video tag.
                        </video>
                    )}
                </div>
            </div>
        </div>
    );
};

const statusFromBackend = (status: BackendAnalysis['status']): ProcessedFile['status'] => {
  if (status === 'processing') return 'Processing';
  if (status === 'completed') return 'Completed';
  return 'Failed';
};

export default function Dashboard() {
  const [analyses, setAnalyses] = useState<ProcessedFile[]>([]);
  const [backendConnected, setBackendConnected] = useState(false);
  const pollingRegister = useRef(new Set<string>());

  const [stats, setStats] = useState<Stats>({
    totalAnalyses: 0,
    processing: 0,
    completed: 0,
    failed: 0,
    avgDexterity: 0,
    systemStatus: 'Offline',
    statusDistribution: [],
    weeklyActivity: [],
  });

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedView, setSelectedView] = useState<'dashboard' | 'analyses' | 'settings'>('dashboard');
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [liveViewAnalysisId, setLiveViewAnalysisId] = useState<string | null>(null);
  const [videoModalData, setVideoModalData] = useState<{ analysisId: string; fileName: string } | null>(null);
  const [isClient, setIsClient] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [sortOption, setSortOption] = useState('date-desc');
  const [filterStatus, setFilterStatus] = useState<'all' | ProcessedFile['status']>('all');
  
  const [analysisSettings, setAnalysisSettings] = useState<AnalysisParameters>({
    min_detection_confidence: 0.75,
    min_tracking_confidence: 0.75,
    filter_min_cutoff: 0.003,
    filter_beta: 0.2,
    outlier_detection_enabled: true,
    outlier_detection_method: 'iqr',
    outlier_threshold: 1.30,
    fingertip_filter_multiplier: 1.0,
    joint_filter_multiplier: 1.0,
    palm_responsiveness_multiplier: 1.0,
  });

  const selectedFile = useMemo(() => analyses.find(a => a.id === selectedFileId) || null, [analyses, selectedFileId]);

  // Load settings from localStorage on component mount
  useEffect(() => {
    setIsClient(true);
    const savedSettings = localStorage.getItem('laixr_analysis_settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        // Merge with defaults to handle new settings that might not be saved
        setAnalysisSettings(prev => ({ ...prev, ...parsed }));
      } catch (error) {
        console.error('Failed to load saved settings:', error);
      }
    }
  }, []);

  // Save settings to localStorage whenever they change
  useEffect(() => {
    if (isClient) {
      localStorage.setItem('laixr_analysis_settings', JSON.stringify(analysisSettings));
    }
  }, [analysisSettings, isClient]);

  const resetSettingsToDefaults = () => {
    const defaults: AnalysisParameters = {
      min_detection_confidence: 0.75,
      min_tracking_confidence: 0.75,
      filter_min_cutoff: 0.003,
      filter_beta: 0.2,
      outlier_detection_enabled: true,
      outlier_detection_method: 'iqr',
      outlier_threshold: 1.30,
      fingertip_filter_multiplier: 1.0,
      joint_filter_multiplier: 1.0,
      palm_responsiveness_multiplier: 1.0,
    };
    setAnalysisSettings(defaults);
  };

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Recalculate derived stats whenever analyses change
  useEffect(() => {
    if (analyses.length === 0) return;

    const completedAnalyses = analyses.filter(a => a.status === 'Completed');
    const processing = analyses.filter(a => a.status === 'Processing').length;
    const failed = analyses.filter(a => a.status === 'Failed').length;
    const completed = completedAnalyses.length;

    const avgDexterity = completedAnalyses.reduce((acc, a) => acc + (a.analysisResult?.summary?.overall_dexterity_score || 0), 0) / (completedAnalyses.length || 1);

    const statusDistribution = [
        { name: 'Completed', value: completed, color: 'hsl(var(--chart-2))' },
        { name: 'Processing', value: processing, color: 'hsl(var(--chart-1))' },
        { name: 'Failed', value: failed, color: 'hsl(var(--destructive))' },
    ].filter(item => item.value > 0);

    const today = new Date();
    const weeklyActivity = Array.from({ length: 7 }).map((_, i) => {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const dateString = d.toLocaleDateString('en-US', { weekday: 'short' });
        const count = analyses.filter(a => new Date(a.date).toDateString() === d.toDateString()).length;
        return { date: dateString, count };
    }).reverse();

    setStats(prev => ({
        ...prev,
        totalAnalyses: analyses.length,
        processing,
        completed,
        failed,
        avgDexterity,
        statusDistribution,
        weeklyActivity,
    }));
  }, [analyses]);

  const sortedAndFilteredAnalyses = useMemo(() => {
    return analyses
      .filter(a => {
        const matchesSearch = a.name.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesStatus = filterStatus === 'all' || a.status === filterStatus;
        return matchesSearch && matchesStatus;
      })
      .sort((a, b) => {
        switch (sortOption) {
          case 'date-asc':
            return new Date(a.date).getTime() - new Date(b.date).getTime();
          case 'score-desc':
            return (b.analysisResult?.summary?.overall_dexterity_score ?? -1) - (a.analysisResult?.summary?.overall_dexterity_score ?? -1);
          case 'score-asc':
            return (a.analysisResult?.summary?.overall_dexterity_score ?? -1) - (b.analysisResult?.summary?.overall_dexterity_score ?? -1);
          case 'date-desc':
          default:
            return new Date(b.date).getTime() - new Date(a.date).getTime();
        }
      });
  }, [analyses, searchQuery, filterStatus, sortOption]);

  const pollAnalysisProgress = useCallback(async (analysisId: string) => {
    if (!analysisId || pollingRegister.current.has(analysisId)) return;
    pollingRegister.current.add(analysisId);

    const poll = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/analysis/${analysisId}`);
        if (!response.ok) {
            pollingRegister.current.delete(analysisId);
            setAnalyses(prev => prev.map(a => a.id === analysisId ? { ...a, status: 'Failed', errorMessage: `Polling failed with status ${response.status}` } : a));
            return;
        }

        const data: BackendAnalysis = await response.json();
        
        setAnalyses(prev => prev.map(a => {
            if (a.id === analysisId) {
                return {
                    ...a,
                    status: statusFromBackend(data.status),
                    errorMessage: data.error_message,
                    analysisResult: data.results ? (typeof data.results === 'string' ? JSON.parse(data.results) : data.results) : undefined,
                };
            }
            return a;
        }));

        if (data.status === 'completed' || data.status === 'failed') {
            pollingRegister.current.delete(analysisId);
            if(liveViewAnalysisId === analysisId) {
                // If we are currently viewing this analysis, let the websocket handle closing
            }
        } else {
            setTimeout(poll, 2000);
        }
      } catch (error) {
        console.error(`Polling error for ${analysisId}:`, error);
        pollingRegister.current.delete(analysisId);
      }
    };
    setTimeout(poll, 1000);
  }, [liveViewAnalysisId]);

  const loadBackendAnalyses = useCallback(async () => {
    setIsRefreshing(true);
    try {
        const response = await fetch(`${BACKEND_URL}/api/analyses`);
        if (!response.ok) throw new Error("Failed to fetch analyses");

        const backendAnalyses: BackendAnalysis[] = await response.json();
        const processedAnalyses = backendAnalyses.map(b => {
            const newFile: ProcessedFile = {
                id: b.id,
                name: b.original_name,
                date: new Date(b.created_at).toISOString(),
                status: statusFromBackend(b.status),
                errorMessage: b.error_message,
                analysisParameters: b.analysis_parameters ? (typeof b.analysis_parameters === 'string' ? JSON.parse(b.analysis_parameters) : b.analysis_parameters) : undefined,
                analysisResult: b.results ? (typeof b.results === 'string' ? JSON.parse(b.results) : b.results) : undefined,
            };
            if (newFile.status === 'Processing' && !pollingRegister.current.has(newFile.id)) {
                pollAnalysisProgress(newFile.id);
            }
            return newFile;
        });
        
        setAnalyses(processedAnalyses);
    } catch (error) {
        console.error("Failed to load backend analyses:", error);
    } finally {
        setIsRefreshing(false);
    }
  }, [pollAnalysisProgress]);

  const checkBackendConnection = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/health`);
      if (response.ok) {
        setBackendConnected(true);
        setStats(prev => ({ ...prev, systemStatus: 'Online' }));
      } else {
        throw new Error('Backend not responding');
      }
    } catch (error) {
      setBackendConnected(false);
      setStats(prev => ({ ...prev, systemStatus: 'Offline' }));
    }
  }, []);

  useEffect(() => {
    checkBackendConnection();
    loadBackendAnalyses();
    const interval = setInterval(checkBackendConnection, 30000);
    return () => clearInterval(interval);
  }, [checkBackendConnection, loadBackendAnalyses]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const tempId = `local-${Date.now()}`;
      const newAnalysis: ProcessedFile = {
        id: tempId,
        name: file.name,
        date: new Date().toISOString(),
        status: 'Uploading',
      };

      setAnalyses(prev => [newAnalysis, ...prev]);

      const formData = new FormData();
      formData.append('file', file);
      Object.entries(analysisSettings).forEach(([key, value]) => {
        formData.append(key, String(value));
      });
      
      try {
        const response = await fetch(`${BACKEND_URL}/api/upload`, {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(result.message || 'Upload failed');
        }
        
        const analysisId = result.analysis_id;

        setAnalyses(prev => prev.map(a => 
          a.id === tempId ? { ...a, status: 'Processing', id: analysisId } : a
        ));
        
        pollAnalysisProgress(analysisId);

      } catch (error) {
        console.error('Upload error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown upload error';
        setAnalyses(prev => prev.map(a => 
          a.id === tempId ? { ...a, status: 'Failed', errorMessage } : a
        ));
      }
    }
  }, [analysisSettings, pollAnalysisProgress]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, disabled: !backendConnected });

  const handleSelectAnalysisForReport = async (analysis: ProcessedFile) => {
    setSelectedFileId(analysis.id);
    setSelectedView('analyses');
  };

  const handleDeleteAnalysis = async (analysisId: string) => {
    try {
        await fetch(`${BACKEND_URL}/api/analysis/${analysisId}`, { method: 'DELETE' });
        loadBackendAnalyses();
    } catch (error) {
        console.error("Failed to delete analysis:", error);
    }
  };

  const handleWatchVideo = (analysisId: string, fileName: string) => {
    setVideoModalData({ analysisId, fileName });
  };

  const handleSettingChange = (param: keyof AnalysisParameters, value: number | string | boolean) => {
    setAnalysisSettings(prev => {
      // Handle different types of settings
      if (param === 'outlier_detection_enabled') {
        return { ...prev, [param]: Boolean(value) };
      } else if (param === 'outlier_detection_method') {
        // Convert numeric selection back to string
        const method = typeof value === 'number' ? (value === 0 ? 'iqr' : 'zscore') : value;
        return { ...prev, [param]: method as 'iqr' | 'zscore' };
      } else {
        // Numeric parameters
        return { ...prev, [param]: Number(value) };
      }
    });
  };

  return (
    <div className="flex h-screen bg-background text-foreground font-sans">
      {liveViewAnalysisId && <LiveAnalysisViewer analysisId={liveViewAnalysisId} onClose={() => {
          setLiveViewAnalysisId(null);
          loadBackendAnalyses();
      }} />}
      
      {videoModalData && <VideoModal 
          analysisId={videoModalData.analysisId} 
          fileName={videoModalData.fileName} 
          onClose={() => setVideoModalData(null)} 
      />}
      
      <nav className="w-64 border-r border-border p-6 flex flex-col justify-between bg-secondary/30">
          <div>
              <div className="flex items-center gap-2 mb-10">
                  <h1 className="text-xl font-bold">LAIXR Handtracking</h1>
              </div>
              <ul>
                  {[
                      { icon: BarChart3, label: 'dashboard' },
                      { icon: FileText, label: 'analyses' },
                      { icon: Settings, label: 'settings' }
                  ].map(item => (
                      <li key={item.label}>
                          <button 
                              onClick={() => setSelectedView(item.label as any)}
                              className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-left transition-colors ${selectedView === item.label ? 'bg-primary/90 text-primary-foreground' : 'hover:bg-primary/10'}`}
                          >
                              <item.icon className="w-5 h-5"/>
                              <span className="capitalize">{item.label}</span>
                          </button>
                      </li>
                  ))}
              </ul>
          </div>
          <div className={`text-sm flex items-center gap-2 p-2 rounded-md ${backendConnected ? 'bg-green-500/10 text-green-700' : 'bg-red-500/10 text-red-700'}`}>
              <div className={`w-2 h-2 rounded-full ${backendConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              Backend {stats.systemStatus}
          </div>
      </nav>
      
      <main className="flex-1 p-8 flex flex-col overflow-y-auto">
        <header className="flex items-center justify-between mb-8">
            <h2 className="text-3xl font-bold capitalize">{selectedView}</h2>
            <div className="flex items-center space-x-4">
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500"/>
                    <input 
                        type="text" 
                        placeholder="Search analyses..." 
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        className="bg-card border border-border rounded-md pl-10 pr-4 py-2 w-64"
                    />
                </div>
                <button onClick={loadBackendAnalyses} disabled={isRefreshing} className="p-2 bg-card rounded-md hover:bg-primary/10 border border-border disabled:opacity-50">
                    <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
                </button>
            </div>
        </header>

        {selectedView === 'dashboard' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
              <div className="lg:col-span-3 grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="bg-card border-border rounded-lg p-6 flex items-center space-x-4"><div className="p-3 rounded-md bg-primary/10"><Database className="w-6 h-6 text-primary"/></div><div><p className="text-muted-foreground text-sm">Total Analyses</p><p className="text-2xl font-bold">{stats.totalAnalyses}</p></div></div>
                <div className="bg-card border-border rounded-lg p-6 flex items-center space-x-4"><div className="p-3 rounded-md bg-green-500/10"><ThumbsUp className="w-6 h-6 text-green-500"/></div><div><p className="text-muted-foreground text-sm">Completed</p><p className="text-2xl font-bold">{stats.completed}</p></div></div>
                <div className="bg-card border-border rounded-lg p-6 flex items-center space-x-4"><div className="p-3 rounded-md bg-blue-500/10"><Clock className="w-6 h-6 text-blue-500"/></div><div><p className="text-muted-foreground text-sm">Processing</p><p className="text-2xl font-bold">{stats.processing}</p></div></div>
                <div className="bg-card border-border rounded-lg p-6 flex items-center space-x-4"><div className="p-3 rounded-md bg-red-500/10"><XCircle className="w-6 h-6 text-red-500"/></div><div><p className="text-muted-foreground text-sm">Failed</p><p className="text-2xl font-bold">{stats.failed}</p></div></div>
              </div>

              <div className="lg:col-span-2 bg-card border-border rounded-lg p-6">
                  <h3 className="text-xl font-semibold mb-4 text-center">Upload New Video</h3>
                  <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'}`}>
                      <input {...getInputProps()} />
                      <Upload className="mx-auto h-10 w-10 text-muted-foreground mb-3" />
                      <p className="text-base mb-1">Drag & drop videos here</p>
                      <p className="text-sm text-muted-foreground">or click to select files</p>
                  </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                <div className="lg:col-span-3 bg-card border-border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><Activity className="w-5 h-5"/>Weekly Analysis Activity</h3>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={stats.weeklyActivity} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <XAxis dataKey="date" tickLine={false} axisLine={false} tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }} />
                            <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}/>
                            <RechartsTooltip
                                cursor={{ fill: 'hsl(var(--primary) / 0.1)' }}
                                contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))' }}
                            />
                            <Bar dataKey="count" name="Analyses" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
                <div className="lg:col-span-2 bg-card border-border rounded-lg p-6 flex flex-col">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><ListChecks className="w-5 h-5"/>Analysis Status</h3>
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={stats.statusDistribution}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                outerRadius="80%"
                                innerRadius="50%"
                                paddingAngle={5}
                            >
                                {stats.statusDistribution.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Pie>
                             <RechartsTooltip
                                contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))' }}
                            />
                            <Legend iconType="circle" />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="bg-card border-border rounded-lg p-6">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold">Recent Analyses</h3>
                    <button onClick={() => setSelectedView('analyses')} className="text-sm text-primary hover:underline">View All</button>
                </div>
                <div className="space-y-3">
                    {sortedAndFilteredAnalyses.length > 0 ? (
                        sortedAndFilteredAnalyses.slice(0, 5).map((file) => (
                          <div key={file.id} className="bg-secondary/40 p-3 rounded-lg flex items-center justify-between hover:bg-secondary/80 transition-colors duration-200">
                            <div className="flex items-center gap-4">
                                <FileVideo className="w-6 h-6 text-primary" />
                                <div>
                                    <p className="font-medium text-sm truncate w-48" title={file.name}>{file.name}</p>
                                    <p className="text-xs text-muted-foreground">{isClient ? new Date(file.date).toLocaleDateString() : ''}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                {getStatusChip(file)}
                                <button onClick={() => handleSelectAnalysisForReport(file)} disabled={file.status !== 'Completed'} className="p-1 disabled:opacity-30 disabled:cursor-not-allowed">
                                    <ChevronRight className="h-5 w-5" />
                                </button>
                            </div>
                          </div>
                        ))
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">No analyses found.</div>
                    )}
                </div>
            </div>
          </div>
        )}

        {selectedView === 'analyses' && !selectedFile && (
           <div className="space-y-6">
            <div className="bg-card border-border rounded-lg p-4 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-2 flex-wrap">
                <p className="font-semibold text-sm mr-2">Status:</p>
                {(['all', 'Completed', 'Processing', 'Failed'] as const).map(status => (
                  <button
                    key={status}
                    onClick={() => setFilterStatus(status)}
                    className={`px-3 py-1 text-xs rounded-full transition-colors ${filterStatus === status ? 'bg-primary text-primary-foreground' : 'bg-secondary/60 hover:bg-secondary'}`}
                  >
                    {status}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <label htmlFor="sort-select" className="font-semibold text-sm">Sort by:</label>
                <select 
                  id="sort-select"
                  value={sortOption}
                  onChange={e => setSortOption(e.target.value)}
                  className="bg-secondary/60 border border-border rounded-md px-3 py-1.5 text-sm"
                >
                  <option value="date-desc">Newest First</option>
                  <option value="date-asc">Oldest First</option>
                  <option value="score-desc">Highest Score</option>
                  <option value="score-asc">Lowest Score</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {sortedAndFilteredAnalyses.map(file => (
                <AnalysisCard
                  key={file.id}
                  file={file}
                  onDelete={handleDeleteAnalysis}
                  onViewReport={handleSelectAnalysisForReport}
                  onViewLive={setLiveViewAnalysisId}
                  onWatchVideo={handleWatchVideo}
                  isClient={isClient}
                />
              ))}
            </div>
            
            {sortedAndFilteredAnalyses.length === 0 && (
              <div className="text-center py-16 text-muted-foreground bg-card rounded-lg">
                  <h3 className="text-xl font-semibold">No Matching Analyses</h3>
                  <p className="mt-2">Try adjusting your search or filter options.</p>
              </div>
            )}
          </div>
        )}

        {selectedView === 'analyses' && selectedFile && (
          <div>
              <button onClick={() => setSelectedFileId(null)} className="flex items-center gap-2 mb-4 text-sm hover:text-primary">
                  <ChevronRight className="w-4 h-4 rotate-180"/> Back to All Analyses
              </button>
              <AnalysisReport file={selectedFile} />
          </div>
        )}

        {selectedView === 'settings' && (
            <div className="max-w-4xl mx-auto">
              <div className="mb-10 text-center">
                <h2 className="text-3xl font-bold tracking-tight">Fine-Tune Your Analysis</h2>
                <p className="mt-2 text-lg text-muted-foreground">Adjust the parameters to optimize hand tracking for your specific videos.</p>
                <button
                  onClick={resetSettingsToDefaults}
                  className="mt-4 px-4 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors"
                >
                  Reset to Defaults
                </button>
              </div>

              <div className="space-y-12">
                <div className="bg-card p-8 rounded-xl border border-border shadow-sm">
                    <div className="flex items-start gap-4 mb-6">
                      <div className="p-3 rounded-lg bg-primary/10">
                        <Search className="w-6 h-6 text-primary"/>
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold">Detection & Tracking</h3>
                        <p className="text-muted-foreground mt-1">Configure how the model identifies and follows hands.</p>
                      </div>
                    </div>
                    <div className="space-y-8 pl-16">
                        <SettingSlider 
                          label="Minimum Detection Confidence"
                          description="The confidence score required for a hand to be initially detected. Higher values reduce false positives."
                          value={analysisSettings.min_detection_confidence}
                          onValueChange={(val) => handleSettingChange('min_detection_confidence', val[0])}
                          min={0.1} max={1.0} step={0.05}
                        />
                        <SettingSlider 
                          label="Minimum Tracking Confidence"
                          description="The confidence score required to continue tracking a hand. Higher values can prevent tracking a wrong object."
                          value={analysisSettings.min_tracking_confidence}
                          onValueChange={(val) => handleSettingChange('min_tracking_confidence', val[0])}
                          min={0.1} max={1.0} step={0.05}
                        />
                    </div>
                </div>
                
                <div className="bg-card p-8 rounded-xl border border-border shadow-sm">
                    <div className="flex items-start gap-4 mb-6">
                      <div className="p-3 rounded-lg bg-primary/10">
                        <Activity className="w-6 h-6 text-primary"/>
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold">Smoothing & Filtering</h3>
                        <p className="text-muted-foreground mt-1">Adjust the One-Euro filter to balance smoothness and responsiveness.</p>
                      </div>
                    </div>
                    <div className="space-y-8 pl-16">
                        <SettingSlider 
                          label="Jitter Reduction (Min Cutoff)"
                          description="Lower values increase smoothing but can also increase lag. Good for slow, precise movements."
                          value={analysisSettings.filter_min_cutoff}
                          onValueChange={(val) => handleSettingChange('filter_min_cutoff', val[0])}
                          min={0.0001} max={0.1} step={0.0001} format="0.0000"
                        />
                        <SettingSlider 
                          label="Responsiveness (Beta)"
                          description="Higher values make the filter react faster to quick movements, reducing lag but potentially increasing jitter."
                          value={analysisSettings.filter_beta}
                          onValueChange={(val) => handleSettingChange('filter_beta', val[0])}
                          min={0.0} max={1.0} step={0.05}
                        />
                    </div>
                </div>

                <div className="bg-card p-8 rounded-xl border border-border shadow-sm">
                    <div className="flex items-start gap-4 mb-6">
                      <div className="p-3 rounded-lg bg-primary/10">
                        <AlertCircle className="w-6 h-6 text-primary"/>
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold">Outlier Detection</h3>
                        <p className="text-muted-foreground mt-1">Automatically detect and remove measurement errors from jerk calculations.</p>
                      </div>
                    </div>
                    <div className="space-y-8 pl-16">
                        <div>
                          <label className="flex items-center gap-3 mb-4">
                            <input
                              type="checkbox"
                              checked={analysisSettings.outlier_detection_enabled}
                              onChange={(e) => handleSettingChange('outlier_detection_enabled', e.target.checked ? 1 : 0)}
                              className="w-4 h-4 rounded border-border"
                            />
                            <span className="font-medium">Enable Outlier Detection</span>
                          </label>
                          <p className="text-sm text-muted-foreground mb-4">Remove extreme jerk values that may be caused by tracking errors.</p>
                        </div>
                        
                        {analysisSettings.outlier_detection_enabled && (
                          <>
                            <div>
                              <label className="block mb-1.5 font-medium text-base">Detection Method</label>
                              <p className="text-sm text-muted-foreground mb-4">Choose the statistical method for outlier detection.</p>
                              <select
                                value={analysisSettings.outlier_detection_method}
                                onChange={(e) => handleSettingChange('outlier_detection_method', e.target.value === 'iqr' ? 0 : 1)}
                                className="w-full p-2 border border-border rounded-md bg-background"
                              >
                                <option value="iqr">IQR (Interquartile Range) - Recommended</option>
                                <option value="zscore">Z-Score Method</option>
                              </select>
                            </div>
                            
                            <SettingSlider 
                              label="Outlier Sensitivity"
                              description="Lower values detect more outliers. 1.5 is aggressive, 2.5 is conservative."
                              value={analysisSettings.outlier_threshold}
                              onValueChange={(val) => handleSettingChange('outlier_threshold', val[0])}
                              min={1.0} max={3.0} step={0.1}
                            />
                          </>
                        )}
                    </div>
                </div>

                <div className="bg-card p-8 rounded-xl border border-border shadow-sm">
                    <div className="flex items-start gap-4 mb-6">
                      <div className="p-3 rounded-lg bg-primary/10">
                        <Settings className="w-6 h-6 text-primary"/>
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold">Advanced Filtering</h3>
                        <p className="text-muted-foreground mt-1">Fine-tune filtering behavior for different hand parts.</p>
                      </div>
                    </div>
                    <div className="space-y-8 pl-16">
                        <SettingSlider 
                          label="Fingertip Stability Multiplier"
                          description="Higher values make fingertips more stable (less jittery). 1.0 = normal, 2.0 = extra stable."
                          value={analysisSettings.fingertip_filter_multiplier}
                          onValueChange={(val) => handleSettingChange('fingertip_filter_multiplier', val[0])}
                          min={0.1} max={3.0} step={0.1}
                        />
                        <SettingSlider 
                          label="Joint Stability Multiplier" 
                          description="Higher values make finger joints more stable. 1.0 = normal, 2.0 = extra stable."
                          value={analysisSettings.joint_filter_multiplier}
                          onValueChange={(val) => handleSettingChange('joint_filter_multiplier', val[0])}
                          min={0.1} max={3.0} step={0.1}
                        />
                        <SettingSlider 
                          label="Palm Responsiveness Multiplier"
                          description="Higher values make palm/wrist tracking more responsive to movement. 1.0 = normal, 2.0 = extra responsive."
                          value={analysisSettings.palm_responsiveness_multiplier}
                          onValueChange={(val) => handleSettingChange('palm_responsiveness_multiplier', val[0])}
                          min={0.1} max={3.0} step={0.1}
                        />
                    </div>
                </div>
              </div>
            </div>
        )}
      </main>
    </div>
  );
}

function SettingSlider({ label, description, value, onValueChange, min, max, step, format="0.00" }: { 
  label: string, 
  description: string,
  value: number, 
  onValueChange: (value: number[]) => void, 
  min: number, 
  max: number, 
  step: number,
  format?: string
}) {
  return (
    <div>
      <label className="block mb-1.5 font-medium text-base">{label}</label>
      <p className="text-sm text-muted-foreground mb-4">{description}</p>
      <div className="flex items-center gap-4">
        <Slider
          min={min}
          max={max}
          step={step}
          value={[value]}
          onValueChange={onValueChange}
        />
        <span className="font-mono text-sm w-24 text-center bg-secondary py-1.5 rounded-md">{value.toFixed(format.split('.')[1]?.length || 2)}</span>
      </div>
    </div>
  )
}

function AnalysisCard({ file, onDelete, onViewReport, onViewLive, onWatchVideo, isClient }: {
  file: ProcessedFile;
  onDelete: (id: string) => void;
  onViewReport: (file: ProcessedFile) => void;
  onViewLive: (id: string) => void;
  onWatchVideo: (analysisId: string, fileName: string) => void;
  isClient: boolean;
}) {
  const { id, name, date, status, analysisResult, errorMessage } = file;

  const summary = analysisResult?.summary;
  const score = summary?.overall_dexterity_score;
  const leftSummary = summary?.Left;
  const rightSummary = summary?.Right;
  
  const totalPath = (leftSummary?.total_path_pixels || 0) + (rightSummary?.total_path_pixels || 0);

  let smoothnessScore = null;
  if (leftSummary?.log_dimensionless_jerk && rightSummary?.log_dimensionless_jerk) {
    smoothnessScore = (leftSummary.log_dimensionless_jerk + rightSummary.log_dimensionless_jerk) / 2;
  } else {
    smoothnessScore = leftSummary?.log_dimensionless_jerk || rightSummary?.log_dimensionless_jerk;
  }
  
  return (
    <div className="bg-card border-border rounded-lg p-5 flex flex-col justify-between shadow-sm hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
      <div>
        <div className="flex justify-between items-start mb-3">
          <FileVideo className="w-8 h-8 text-primary flex-shrink-0 mr-4" />
          {getStatusChip(file)}
        </div>
        <h3 className="font-bold text-lg truncate mb-1" title={name}>{name}</h3>
        <p className="text-xs text-muted-foreground mb-4">
          {isClient ? new Date(date).toLocaleString() : 'Loading date...'}
        </p>
        
        {status === 'Completed' && summary && (
          <div className="space-y-3 text-sm mb-4 border-t border-border pt-4">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground flex items-center gap-1.5"><Gauge className="w-4 h-4"/> Dexterity Score</span>
              <span className="font-semibold">{score?.toFixed(1) ?? 'N/A'} / 100</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground flex items-center gap-1.5"><Sparkles className="w-4 h-4"/> Smoothness</span>
              <span className="font-semibold">{smoothnessScore?.toFixed(2) ?? 'N/A'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground flex items-center gap-1.5"><ArrowRightLeft className="w-4 h-4"/> Path Length</span>
              <span className="font-semibold">{totalPath.toFixed(0)} px</span>
            </div>
          </div>
        )}
        
        {status === 'Failed' && (
           <div className="text-sm my-4 p-3 bg-destructive/10 rounded-md text-destructive">
             <p className="font-semibold">Analysis Failed</p>
             <p className="truncate">{errorMessage}</p>
           </div>
        )}
      </div>

      <div className="border-t border-border pt-4 flex items-center justify-end gap-2">
        <TooltipProvider>
          <Tooltip><TooltipTrigger asChild>
              <button onClick={() => onViewLive(id)} disabled={status !== 'Processing'} className="p-2 rounded-md hover:bg-primary/10 disabled:opacity-30 disabled:cursor-not-allowed"><Eye className="w-4 h-4"/></button>
          </TooltipTrigger><TooltipContent><p>View Live Feed</p></TooltipContent></Tooltip>
          <Tooltip><TooltipTrigger asChild>
            <button onClick={() => onWatchVideo(id, name)} disabled={status !== 'Completed'} className="p-2 rounded-md hover:bg-primary/10 disabled:opacity-30 disabled:cursor-not-allowed"><Play className="w-4 h-4"/></button>
          </TooltipTrigger><TooltipContent><p>Watch Video</p></TooltipContent></Tooltip>
          <Tooltip><TooltipTrigger asChild>
            <button onClick={() => onViewReport(file)} disabled={status !== 'Completed'} className="p-2 rounded-md hover:bg-primary/10 disabled:opacity-30 disabled:cursor-not-allowed"><BarChart3 className="w-4 h-4"/></button>
          </TooltipTrigger><TooltipContent><p>View Full Report</p></TooltipContent></Tooltip>
          <Tooltip><TooltipTrigger asChild>
            <button onClick={() => onDelete(id)} className="p-2 rounded-md hover:bg-destructive/10 text-destructive"><X className="w-4 h-4"/></button>
          </TooltipTrigger><TooltipContent><p>Delete Analysis</p></TooltipContent></Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}
