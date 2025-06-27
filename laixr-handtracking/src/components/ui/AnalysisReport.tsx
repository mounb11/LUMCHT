'use client';

import React, { useRef, useState, useMemo, useCallback, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, Brush } from 'recharts';
import { Download, Trash2, Loader2, AlertCircle, Hand, Eye, RefreshCw, EyeOff, CheckCircle, Activity, Play, Pause, Volume2, VolumeX, Maximize, RotateCcw } from 'lucide-react';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// --- Performance Optimization Hook ---
// This hook creates a debounced version of a callback.
// It delays invoking the function until after `delay` ms have elapsed
// since the last time the debounced function was invoked.
// This is crucial for performance when dealing with rapidly firing events, like a slider being dragged.
function useDebouncedCallback<T extends (...args: any[]) => any>(
    callback: T,
    delay: number
) {
    const callbackRef = useRef(callback);
    useEffect(() => {
        callbackRef.current = callback;
    }, [callback]);

    return useMemo(() => {
        const debounced = (...args: Parameters<T>) => {
            const timer = setTimeout(() => {
                callbackRef.current(...args);
            }, delay);

            return () => {
                clearTimeout(timer);
            };
        };
        // This is a simplified version that should avoid the complex type issues
        // It doesn't clear previous timeouts but for this UI it will be sufficient
        // to prevent the jank, which is the main goal.
        let timeout: ReturnType<typeof setTimeout>;
        return (...args: Parameters<T>): void => {
            clearTimeout(timeout);
            timeout = setTimeout(() => callbackRef.current(...args), delay);
        }
    }, [delay]);
}

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  className?: string;
}

const MetricCard = ({ label, value, unit, className }: MetricCardProps) => (
  <div className={`bg-secondary/50 p-4 rounded-lg ${className}`}>
    <p className="text-sm text-muted-foreground mb-1">{label}</p>
    <p className="text-2xl font-semibold">
      {typeof value === 'number' ? value.toFixed(2) : value}
      {unit && <span className="text-base text-muted-foreground ml-1">{unit}</span>}
    </p>
  </div>
);

interface BrushState {
  startIndex: number;
  endIndex: number;
}

interface TimeRange {
  from: number;
  to: number;
}

// Optimize data by sampling for better performance
const sampleData = (data: any[], maxPoints: number = 500): any[] => {
  if (!data || data.length <= maxPoints) return data;
  
  const step = Math.ceil(data.length / maxPoints);
  const sampled = [];
  
  for (let i = 0; i < data.length; i += step) {
    sampled.push(data[i]);
  }
  
  // Always include the last point
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }
  
  return sampled;
};

function KinematicChartWithBrush({ 
    data, 
    dataKey, 
    title, 
    color, 
    unit,
    showBrush = false,
    brushSelection,
    onBrushChange,
    previewMode = false,
    previewRanges = []
}: {
  data: any[];
  dataKey: string;
  title: string;
  color: string;
  unit: string;
  showBrush?: boolean;
  brushSelection?: { startIndex: number; endIndex: number } | null;
  onBrushChange?: (selection: { startIndex: number; endIndex: number } | null) => void;
  previewMode?: boolean;
  previewRanges?: TimeRange[];
}) {
    if (!data || data.length === 0) {
        return <div className="p-4 text-center bg-secondary rounded-lg h-80 flex items-center justify-center">No data available for {title}.</div>;
    }
    
    // Sample data for performance
    const sampledData = useMemo(() => sampleData(data, 300), [data]);
    const brushData = useMemo(() => sampleData(data, 100), [data]);
    
    // Filter out preview ranges if in preview mode
    const displayData = useMemo(() => {
        if (!previewMode || previewRanges.length === 0) return sampledData;
        
        return sampledData.map(point => {
            const inPreviewRange = previewRanges.some(range => 
                point.time >= range.from && point.time <= range.to
            );
            
            return {
                ...point,
                [dataKey]: inPreviewRange ? null : point[dataKey],
                preview: inPreviewRange
            };
        });
    }, [sampledData, previewMode, previewRanges, dataKey]);
    
    // Convert brush selection to sampled indices if needed
    const sampledBrushSelection = useMemo(() => {
        if (!brushSelection) return null;
        
        // Map original indices to sampled indices
        const originalToSampled = (originalIndex: number) => {
            const ratio = data.length / sampledData.length;
            return Math.round(originalIndex / ratio);
        };
        
        return {
            startIndex: originalToSampled(brushSelection.startIndex),
            endIndex: originalToSampled(brushSelection.endIndex)
        };
    }, [brushSelection, data.length, sampledData.length]);
    
    const handleBrushChange = useCallback((domain: any) => {
        if (!onBrushChange || !domain) return;
        
        const { startIndex, endIndex } = domain;
        if (startIndex === undefined || endIndex === undefined) {
            onBrushChange(null);
            return;
        }
        
        // Map sampled indices back to original indices
        const sampledToOriginal = (sampledIndex: number) => {
            const ratio = data.length / sampledData.length;
            return Math.round(sampledIndex * ratio);
        };
        
        onBrushChange({
            startIndex: sampledToOriginal(startIndex),
            endIndex: sampledToOriginal(endIndex)
        });
    }, [onBrushChange, data.length, sampledData.length]);
    
    // Calculate chart domain for proper scaling
    const chartDomain = useMemo(() => {
        if (!displayData || displayData.length === 0) return ['dataMin', 'dataMax'];
        const values = displayData.map(d => d[dataKey]).filter(v => v !== null && v !== undefined);
        if (values.length === 0) return ['dataMin', 'dataMax'];
        
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal;
        const padding = range * 0.1; // 10% padding
        
        return [Math.max(0, minVal - padding), maxVal + padding];
    }, [displayData, dataKey]);
    
    return (
        <div className="bg-secondary/50 rounded-lg p-4 h-80">
            <h4 className="font-semibold text-center mb-4">{title} ({unit})</h4>
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart 
                    data={displayData} 
                    margin={{ top: 5, right: 20, left: 10, bottom: showBrush ? 50 : 20 }}
                >
                    <defs>
                        <linearGradient id={`color-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={color} stopOpacity={0.4}/>
                            <stop offset="95%" stopColor={color} stopOpacity={0.1}/>
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted-foreground) / 0.2)" />
                    <XAxis 
                        dataKey="time" 
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(time) => `${time.toFixed(2)}s`}
                        label={{ value: "Time (s)", position: "insideBottom", offset: -15 }}
                    />
                    <YAxis 
                        domain={chartDomain}
                        label={{ value: unit, angle: -90, position: 'insideLeft', offset: 10 }}
                        tickFormatter={(value) => value.toExponential(1)}
                        width={80}
                    />
                    <Tooltip
                        contentStyle={{ 
                            backgroundColor: 'hsl(var(--background))', 
                            borderColor: 'hsl(var(--border))' 
                        }}
                        labelFormatter={(time) => `Time: ${time.toFixed(2)}s`}
                        formatter={(value: number, name: string) => {
                            if (value === null) return ['Removed (preview)', name];
                            return [`${value.toFixed(2)} ${unit}`, name];
                        }}
                    />
                    
                    {/* Show preview removal regions */}
                    {previewMode && previewRanges.map((range, idx) => {
                        const xStart = ((range.from - (displayData[0]?.time || 0)) / ((displayData[displayData.length - 1]?.time || 1) - (displayData[0]?.time || 0))) * 100;
                        const xWidth = ((range.to - range.from) / ((displayData[displayData.length - 1]?.time || 1) - (displayData[0]?.time || 0))) * 100;
                        
                        return (
                            <rect
                                key={`preview-${idx}`}
                                x={`${xStart}%`}
                                y="0"
                                width={`${xWidth}%`}
                                height="100%"
                                fill="rgba(239, 68, 68, 0.2)"
                                stroke="rgba(239, 68, 68, 0.6)"
                                strokeWidth="2"
                                strokeDasharray="3 3"
                            />
                        );
                    })}
                    
                    <Area 
                        type="monotone" 
                        dataKey={dataKey} 
                        stroke={color} 
                        fillOpacity={1} 
                        fill={`url(#color-${dataKey})`} 
                        name={title}
                        isAnimationActive={false}
                        connectNulls={!previewMode}
                    />
                    {showBrush && (
                        <Brush
                            dataKey="time"
                            height={30}
                            stroke={color}
                            startIndex={sampledBrushSelection?.startIndex}
                            endIndex={sampledBrushSelection?.endIndex}
                            onChange={handleBrushChange}
                            data={brushData}
                            travellerWidth={8}
                            tickFormatter={(time) => `${time.toFixed(1)}s`}
                        />
                    )}
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}

function HandAnalysisSection({ 
    hand, 
    analysis, 
    summary,
    brushSelection,
    onBrushChange,
    previewMode = false,
    previewRanges = []
}: { 
    hand: 'Left' | 'Right', 
    analysis: any, 
    summary: any,
    brushSelection: BrushState | null,
    onBrushChange: (selection: BrushState | null) => void,
    previewMode?: boolean,
    previewRanges?: TimeRange[]
}) {
    if (!summary || Object.keys(summary).length === 0) {
        return null;
    }

    if (!analysis || !Array.isArray(analysis.jerk) || !Array.isArray(analysis.velocity) || !Array.isArray(analysis.acceleration)) {
        return (
            <div className="space-y-6">
                <h3 className="text-2xl font-bold tracking-tight text-primary">{hand} Hand Analysis</h3>
                <div className="p-4 text-center bg-secondary rounded-lg">Incomplete or old analysis data format. Please re-analyze.</div>
            </div>
        );
    }
    
    // Memoize filtered data to avoid recalculation
    const { jerkData, velocityData, accelerationData, pathData, cumulativeJerkData } = useMemo(() => ({
        jerkData: analysis.jerk.filter((d: any) => d.hand === hand),
        velocityData: analysis.velocity.filter((d: any) => d.hand === hand),
        accelerationData: analysis.acceleration.filter((d: any) => d.hand === hand),
        pathData: analysis.cumulative_path_length.filter((d: any) => d.hand === hand),
        cumulativeJerkData: analysis.cumulative_jerk ? analysis.cumulative_jerk.filter((d: any) => d.hand === hand) : []
    }), [analysis, hand]);

    return (
        <div className="space-y-6">
            <h3 className="text-2xl font-bold tracking-tight text-primary">{hand} Hand Analysis</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Peak Velocity" value={summary.peak_velocity_pps} unit="pps" />
                <MetricCard label="Peak Acceleration" value={summary.peak_acceleration_pps2} unit="pps²" />
                <MetricCard label="Total Path" value={summary.total_path_pixels} unit="pixels" />
                <MetricCard label="Log Smoothness (DJ)" value={summary.log_dimensionless_jerk} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <MetricCard label="Peak Jerk" value={summary.peak_jerk_pps3} unit="pps³" />
                <MetricCard label="Total Cumulative Jerk" value={summary.total_cumulative_jerk} unit="pps³·s" />
                <MetricCard label="Dexterity Score" value={summary.dexterity_score} unit="/ 100" />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <KinematicChartWithBrush 
                    data={velocityData} 
                    dataKey="Velocity" 
                    title="Velocity" 
                    color="var(--chart-1)" 
                    unit="pps"
                    previewMode={previewMode}
                    previewRanges={previewRanges}
                />
                <KinematicChartWithBrush 
                    data={accelerationData} 
                    dataKey="Acceleration" 
                    title="Acceleration" 
                    color="var(--chart-2)" 
                    unit="pps²"
                    previewMode={previewMode}
                    previewRanges={previewRanges}
                />
                <KinematicChartWithBrush 
                    data={jerkData} 
                    dataKey="Jerk" 
                    title="Jerk" 
                    color="var(--chart-3)" 
                    unit="pps³" 
                    showBrush={true}
                    brushSelection={brushSelection}
                    onBrushChange={onBrushChange}
                    previewMode={previewMode}
                    previewRanges={previewRanges}
                />
                {cumulativeJerkData.length > 0 && (
                    <KinematicChartWithBrush 
                        data={cumulativeJerkData} 
                        dataKey="Cumulative Jerk" 
                        title="Cumulative Jerk" 
                        color="var(--chart-5)" 
                        unit="pps³·s"
                        previewMode={previewMode}
                        previewRanges={previewRanges}
                    />
                )}
                <KinematicChartWithBrush 
                    data={pathData} 
                    dataKey="Cumulative Path Length" 
                    title="Cumulative Path Length" 
                    color="var(--chart-6)" 
                    unit="pixels"
                    previewMode={previewMode}
                    previewRanges={previewRanges}
                />
            </div>
        </div>
    );
}

// Video Player Component
const VideoPlayer = ({ analysisId }: { analysisId: string }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const videoUrl = `${BACKEND_URL}/api/analysis/${analysisId}/video`;

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play().catch((err) => {
          setError(`Playback failed: ${err.message}`);
        });
      }
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement && containerRef.current) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else if (document.fullscreenElement) {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const restart = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play().catch((err) => {
        setError(`Playback failed: ${err.message}`);
      });
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

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
    <div className="bg-card rounded-lg p-6 border border-border">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Play className="w-5 h-5 text-primary" />
        Video with Hand Tracking
      </h3>
      
      {error ? (
        <div className="flex flex-col items-center justify-center p-8 bg-destructive/10 rounded-lg text-destructive">
          <AlertCircle className="w-12 h-12 mb-4" />
          <p className="text-center font-semibold">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Reload Page
          </button>
        </div>
      ) : (
        <div ref={containerRef} className={`relative bg-black rounded-lg overflow-hidden ${isFullscreen ? 'w-full h-full' : ''}`}>
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full h-auto max-h-96"
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
            onError={handleVideoError}
            onLoadedData={handleVideoLoad}
            controls={false}
            playsInline
          />
          
          {/* Custom Controls Overlay */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
            <div className="flex items-center justify-between text-white">
              <div className="flex items-center gap-3">
                <button
                  onClick={togglePlay}
                  className="p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
                >
                  {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                </button>
                
                <button
                  onClick={restart}
                  className="p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
                  title="Restart"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                
                <button
                  onClick={toggleMute}
                  className="p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
                >
                  {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                </button>
              </div>
              
              <button
                onClick={toggleFullscreen}
                className="p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
              >
                <Maximize className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default function AnalysisReport({ file }: { file: any }) {
  const [results, setResults] = useState(file.analysisResult);
  const [isCorrecting, setIsCorrecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [leftBrushSelection, setLeftBrushSelection] = useState<BrushState | null>(null);
  const [rightBrushSelection, setRightBrushSelection] = useState<BrushState | null>(null);
  const [previewMode, setPreviewMode] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  if (!results || !results.summary || !results.analysis) {
    return <div className="text-center py-10">No analysis data available or data is in an old format. Please re-analyze the video.</div>;
  }

  const { analysis, summary } = results;
  const leftHandDataExists = summary.Left && Object.keys(summary.Left).length > 0;
  const rightHandDataExists = summary.Right && Object.keys(summary.Right).length > 0;

  const handleDownloadCsv = () => {
    window.location.href = `${BACKEND_URL}/api/analysis/${file.id}/download_csv`;
  };

  // Refresh analysis data from backend
  const refreshAnalysisData = async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/analysis/${file.id}`);
      if (response.ok) {
        const updatedData = await response.json();
        setResults(updatedData);
        // Update the file object as well
        file.analysisResult = updatedData;
        // Force re-render of all components
        setRefreshKey(prev => prev + 1);
      }
    } catch (err) {
      console.error('Failed to refresh analysis data:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleRemoveSelection = async (hand: 'Left' | 'Right') => {
    setError(null);
    const brushSelection = hand === 'Left' ? leftBrushSelection : rightBrushSelection;
    
    if (!brushSelection) {
      setError("Please select a range in the jerk chart to remove.");
      return;
    }

    // Get the jerk data for the selected hand
    const jerkData = analysis.jerk.filter((d: any) => d.hand === hand);
    
    if (!jerkData || jerkData.length === 0) {
      setError("No jerk data available for the selected hand.");
      return;
    }
    
    // Get the time values at the selected indices
    const startTime = jerkData[brushSelection.startIndex]?.time;
    const endTime = jerkData[brushSelection.endIndex]?.time;
    
    if (startTime === undefined || endTime === undefined) {
      setError("Could not determine time range from selection.");
      return;
    }

    setIsCorrecting(true);
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/analysis/${file.id}/correct`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                hand: hand,
                time_ranges_to_remove: [{
                    start_time: startTime,
                    end_time: endTime
                }]
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to correct analysis');
        }
        
        // Get the updated analysis data
        const updatedData = await response.json();
        setResults(updatedData);
        
        // Update the file object as well
        file.analysisResult = updatedData;
        
        // Clear brush selections after successful correction
        if (hand === 'Left') {
            setLeftBrushSelection(null);
        } else {
            setRightBrushSelection(null);
        }
        
        // Turn off preview mode
        setPreviewMode(false);
        
        // Force re-render of all components
        setRefreshKey(prev => prev + 1);

    } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
        setIsCorrecting(false);
    }
  };
  
  // Format selected time range for display
  const formatTimeRange = (hand: 'Left' | 'Right') => {
    const brushSelection = hand === 'Left' ? leftBrushSelection : rightBrushSelection;
    if (!brushSelection) return null;
    
    const jerkData = analysis.jerk.filter((d: any) => d.hand === hand);
    if (!jerkData || jerkData.length === 0) return null;
    
    const startTime = jerkData[brushSelection.startIndex]?.time;
    const endTime = jerkData[brushSelection.endIndex]?.time;
    
    if (startTime === undefined || endTime === undefined) return null;
    
    return `${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`;
  };

  // Get preview ranges for the current selections
  const getPreviewRanges = (hand: 'Left' | 'Right'): TimeRange[] => {
    if (!previewMode) return [];
    
    const brushSelection = hand === 'Left' ? leftBrushSelection : rightBrushSelection;
    if (!brushSelection) return [];
    
    const jerkData = analysis.jerk.filter((d: any) => d.hand === hand);
    if (!jerkData || jerkData.length === 0) return [];
    
    const startTime = jerkData[brushSelection.startIndex]?.time;
    const endTime = jerkData[brushSelection.endIndex]?.time;
    
    if (startTime === undefined || endTime === undefined) return [];
    
    return [{ from: startTime, to: endTime }];
  };

  return (
    <div className="space-y-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
           <MetricCard label="Overall Dexterity Score" value={summary.overall_dexterity_score?.toFixed(2) || 'N/A'} unit="/ 100" className="md:col-span-1 bg-card border" />
           <div className="md:col-span-2 grid grid-cols-1 gap-4">
                <MetricCard label="File Name" value={file.name} className="col-span-2"/>
           </div>
        </div>

        {/* --- Data Correction UI --- */}
        <div className="p-4 border rounded-lg bg-card shadow-sm">
            <h4 className="font-semibold text-lg mb-3 flex items-center"><Hand className="mr-2 h-5 w-5 text-primary" />Data Correction</h4>
            <p className="text-sm text-muted-foreground mb-4">
                Select a range in the jerk chart below to remove outliers caused by tracking errors. 
                Use preview mode to see the effect before applying.
            </p>
            
            {/* Preview Toggle */}
            <div className="mb-4">
                <button
                    onClick={() => setPreviewMode(!previewMode)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                        previewMode ? 'bg-primary text-primary-foreground' : 'bg-secondary'
                    }`}
                >
                    <Eye className="w-4 h-4" />
                    {previewMode ? 'Preview Mode ON' : 'Preview Mode OFF'}
                </button>
            </div>
            
            <div className="flex flex-wrap gap-4">
                {/* Left Hand Selection */}
                {leftHandDataExists && (
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">Left Hand:</span>
                        {leftBrushSelection && formatTimeRange('Left') ? (
                            <>
                                <span className="text-sm bg-secondary px-2 py-1 rounded">
                                    {formatTimeRange('Left')}
                                </span>
                                <button 
                                    onClick={() => handleRemoveSelection('Left')}
                                    disabled={isCorrecting || isRefreshing}
                                    className="flex items-center gap-1 px-3 py-1 bg-destructive text-destructive-foreground rounded-md hover:bg-destructive/90 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isCorrecting ? (
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                    ) : (
                                        <Trash2 className="w-3 h-3" />
                                    )}
                                    Remove
                                </button>
                                <button
                                    onClick={() => setLeftBrushSelection(null)}
                                    className="text-sm text-muted-foreground hover:text-foreground"
                                >
                                    Clear
                                </button>
                            </>
                        ) : (
                            <span className="text-sm text-muted-foreground">No selection</span>
                        )}
                    </div>
                )}
                
                {/* Right Hand Selection */}
                {rightHandDataExists && (
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">Right Hand:</span>
                        {rightBrushSelection && formatTimeRange('Right') ? (
                            <>
                                <span className="text-sm bg-secondary px-2 py-1 rounded">
                                    {formatTimeRange('Right')}
                                </span>
                                <button 
                                    onClick={() => handleRemoveSelection('Right')}
                                    disabled={isCorrecting || isRefreshing}
                                    className="flex items-center gap-1 px-3 py-1 bg-destructive text-destructive-foreground rounded-md hover:bg-destructive/90 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isCorrecting ? (
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                    ) : (
                                        <Trash2 className="w-3 h-3" />
                                    )}
                                    Remove
                                </button>
                                <button
                                    onClick={() => setRightBrushSelection(null)}
                                    className="text-sm text-muted-foreground hover:text-foreground"
                                >
                                    Clear
                                </button>
                            </>
                        ) : (
                            <span className="text-sm text-muted-foreground">No selection</span>
                        )}
                    </div>
                )}
            </div>
            
             {error && (
                <div className="mt-3 text-sm text-destructive-foreground bg-destructive p-3 rounded-lg flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    <p>{error}</p>
                </div>
            )}
        </div>

        <div className="flex justify-end gap-2">
            <button 
                onClick={handleDownloadCsv}
                className="flex items-center gap-2 px-4 py-2 bg-primary/90 text-primary-foreground rounded-md hover:bg-primary transition-colors"
            >
                <Download className="w-4 h-4" />
                Download Raw Landmarks CSV
            </button>
            <button 
                onClick={() => window.location.href = `${BACKEND_URL}/api/analysis/${file.id}/download_timeseries_csv`}
                className="flex items-center gap-2 px-4 py-2 bg-primary/90 text-primary-foreground rounded-md hover:bg-primary transition-colors"
            >
                <Download className="w-4 h-4" />
                Download Kinematics CSV
            </button>
            <button 
                onClick={() => window.location.href = `${BACKEND_URL}/api/analysis/${file.id}/download_final_values_csv`}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
                <Download className="w-4 h-4" />
                Download Final Values CSV
            </button>
        </div>

      <div className="space-y-10 bg-background p-2" key={refreshKey}>
        {leftHandDataExists && (
            <HandAnalysisSection 
                hand="Left" 
                analysis={analysis} 
                summary={summary.Left}
                brushSelection={leftBrushSelection}
                onBrushChange={setLeftBrushSelection}
                previewMode={previewMode}
                previewRanges={getPreviewRanges('Left')}
            />
        )}
        {rightHandDataExists && (
            <HandAnalysisSection 
                hand="Right" 
                analysis={analysis} 
                summary={summary.Right}
                brushSelection={rightBrushSelection}
                onBrushChange={setRightBrushSelection}
                previewMode={previewMode}
                previewRanges={getPreviewRanges('Right')}
            />
        )}
        
        {!leftHandDataExists && !rightHandDataExists && (
            <div className="text-center py-16 bg-card rounded-lg">
                <h3 className="text-xl font-semibold">No Hands Detected</h3>
                <p className="text-muted-foreground mt-2">The analysis could not find any hands in the video.</p>
            </div>
        )}
      </div>

      <VideoPlayer analysisId={file.id} />
    </div>
  );
} 