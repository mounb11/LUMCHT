// Real MediaPipe Hand Tracking Implementation

export interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

export interface HandResults {
  landmarks: HandLandmark[][];
  handedness: string[];
  timestamp: number;
}

export interface PathData {
  timestamp: number;
  pathLength: number;
}

export interface AnalysisResult {
  dexterityScore: number;
  handMovements: number;
  averageSpeed: number;
  tremorDetection: number;
  coordinationScore: number;
  totalPathLength: number;
  pathData: PathData[];
  frameResults: HandResults[];
  processingTime: number;
  videoMetadata: {
    duration: number;
    fps: number;
    frameCount: number;
  };
}

// MediaPipe Global Types
declare global {
  interface Window {
    Hands: any;
    drawConnectors: any;
    drawLandmarks: any;
    HAND_CONNECTIONS: any;
  }
}

class HandAnalysisService {
  private hands: any;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private video: HTMLVideoElement | null = null;
  private mediaPipeLoaded: boolean = false;
  private isProcessing: boolean = false;

  constructor() {
    // Don't initialize DOM elements in constructor to prevent SSR issues
    // They will be initialized when needed
  }

  private initializeDOMElements() {
    if (typeof window === 'undefined') {
      throw new Error('DOM elements can only be initialized in browser environment');
    }
    
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d')!;
    }
    
    if (!this.video) {
      this.video = document.createElement('video');
    }
  }

  private async loadMediaPipe(): Promise<void> {
    if (this.mediaPipeLoaded) return;

    return new Promise((resolve, reject) => {
      // Load MediaPipe scripts in correct order
      const scripts = [
        'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js'
      ];

      const loadScript = (src: string) => {
        return new Promise<void>((resolveScript, rejectScript) => {
          const script = document.createElement('script');
          script.src = src;
          script.onload = () => {
            console.log(`Loaded: ${src}`);
            resolveScript();
          };
          script.onerror = () => rejectScript(new Error(`Failed to load: ${src}`));
          document.head.appendChild(script);
        });
      };

      // Load scripts sequentially
      const loadAllScripts = async () => {
        try {
          for (const script of scripts) {
            await loadScript(script);
          }

          // Wait a bit for MediaPipe to initialize
          await new Promise(resolve => setTimeout(resolve, 1000));

          if (typeof window.Hands === 'undefined') {
            throw new Error('MediaPipe Hands not available after loading');
          }

          // Initialize MediaPipe Hands
          this.hands = new window.Hands({
            locateFile: (file: string) => {
              return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
          });

          this.hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
          });

          this.mediaPipeLoaded = true;
          console.log('MediaPipe Hands initialized successfully');
          resolve();

        } catch (error) {
          console.error('MediaPipe loading failed:', error);
          reject(error);
        }
      };

      loadAllScripts();
    });
  }

  async analyzeVideo(
    videoFile: File,
    onProgress: (progress: number) => void
  ): Promise<AnalysisResult> {
    if (this.isProcessing) {
      throw new Error('Analysis already in progress');
    }

    this.isProcessing = true;
    const startTime = Date.now();

    try {
      // Initialize DOM elements
      this.initializeDOMElements();
      
      // Load MediaPipe first
      await this.loadMediaPipe();

      return new Promise((resolve, reject) => {
        const frameResults: HandResults[] = [];
        let frameCount = 0;
        let processedFrames = 0;

        // Set up video
        this.video!.src = URL.createObjectURL(videoFile);
        this.video!.muted = true;
        this.video!.crossOrigin = 'anonymous';

        this.video!.onloadedmetadata = () => {
          this.canvas!.width = this.video!.videoWidth || 640;
          this.canvas!.height = this.video!.videoHeight || 480;
          
          const duration = this.video!.duration;
          const fps = 10; // Process 10 frames per second for efficiency
          const frameInterval = 1000 / fps;
          const totalFramesToProcess = Math.floor(duration * fps);
          
          console.log(`Analyzing video: ${duration}s, ${totalFramesToProcess} frames to process at ${fps}fps`);

          let currentTime = 0;

          // Set up MediaPipe results handler
          this.hands.onResults((results: any) => {
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
              const landmarks: HandLandmark[][] = results.multiHandLandmarks.map((hand: any) => 
                hand.map((landmark: any) => ({
                  x: landmark.x,
                  y: landmark.y,
                  z: landmark.z
                }))
              );

              const handedness = results.multiHandedness?.map((hand: any) => hand.label) || [];
              
              frameResults.push({
                landmarks,
                handedness,
                timestamp: currentTime
              });
            }

            processedFrames++;
            const progress = totalFramesToProcess > 0 ? Math.round((processedFrames / totalFramesToProcess) * 100) : 0;
            onProgress(progress);
            
            // Continue to next frame
            currentTime += frameInterval / 1000;
            if (currentTime < duration) {
              processFrame();
            } else {
              // Analysis complete
              completeAnalysis();
            }
          });

          const processFrame = () => {
            this.video!.currentTime = currentTime;
            this.video!.onseeked = () => {
              // Draw current frame to canvas
              this.ctx!.drawImage(this.video!, 0, 0, this.canvas!.width, this.canvas!.height);
              
              // Send frame to MediaPipe
              this.hands.send({ image: this.canvas });
              frameCount++;
            };
          };

          const completeAnalysis = () => {
            const processingTime = Date.now() - startTime;
            const result = this.calculateFinalResults(frameResults, processingTime, {
              duration,
              fps,
              frameCount: frameCount
            });
            
            // Cleanup
            URL.revokeObjectURL(this.video!.src);
            this.isProcessing = false;
            
            console.log(`Analysis completed: ${frameResults.length} frames with hands detected out of ${processedFrames} processed`);
            resolve(result);
          };

          // Start processing
          processFrame();
        };

        this.video!.onerror = () => {
          this.isProcessing = false;
          URL.revokeObjectURL(this.video!.src);
          reject(new Error('Failed to load video'));
        };
      });

    } catch (error) {
      this.isProcessing = false;
      throw error;
    }
  }

  private calculateMovementBetweenFrames(prev: HandLandmark[][], curr: HandLandmark[][]): number {
    let totalMovement = 0;
    
    for (let handIndex = 0; handIndex < Math.min(prev.length, curr.length); handIndex++) {
      const prevHand = prev[handIndex];
      const currHand = curr[handIndex];
      
      for (let i = 0; i < Math.min(prevHand.length, currHand.length); i++) {
        const dx = currHand[i].x - prevHand[i].x;
        const dy = currHand[i].y - prevHand[i].y;
        const dz = currHand[i].z - prevHand[i].z;
        totalMovement += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
    }
    
    return totalMovement;
  }

  private calculateFinalResults(frameResults: HandResults[], processingTime: number, metadata: any): AnalysisResult {
    if (frameResults.length === 0) {
      return {
        dexterityScore: 0,
        handMovements: 0,
        averageSpeed: 0,
        tremorDetection: 0,
        coordinationScore: 0,
        totalPathLength: 0,
        pathData: [],
        frameResults: [],
        processingTime,
        videoMetadata: metadata
      };
    }

    // Calculate real metrics based on MediaPipe data
    const dexterityScore = this.calculateDexterityScore(frameResults);
    const handMovements = this.countHandMovements(frameResults);
    const averageSpeed = this.calculateAverageSpeed(frameResults);
    const tremorDetection = this.detectTremor(frameResults);
    const coordinationScore = this.calculateCoordinationScore(frameResults);
    const { totalPathLength, pathData } = this.calculatePathLength(frameResults);

    return {
      dexterityScore: Math.round(dexterityScore * 100) / 100,
      handMovements: Math.round(handMovements),
      averageSpeed: Math.round(averageSpeed * 1000) / 1000,
      tremorDetection: Math.round(tremorDetection * 100) / 100,
      coordinationScore: Math.round(coordinationScore * 100) / 100,
      totalPathLength: Math.round(totalPathLength * 100) / 100,
      pathData,
      frameResults,
      processingTime,
      videoMetadata: metadata
    };
  }

  private calculatePathLength(frameResults: HandResults[]): { totalPathLength: number, pathData: PathData[] } {
    let totalPathLength = 0;
    const pathData: PathData[] = [];

    for (let i = 1; i < frameResults.length; i++) {
      const prev = frameResults[i - 1];
      const curr = frameResults[i];

      if (prev.landmarks.length > 0 && curr.landmarks.length > 0) {
        const movement = this.calculateMovementBetweenFrames(prev.landmarks, curr.landmarks);
        totalPathLength += movement;
        pathData.push({
          timestamp: curr.timestamp,
          pathLength: totalPathLength
        });
      }
    }
    return { totalPathLength, pathData };
  }

  private calculateDexterityScore(frameResults: HandResults[]): number {
    let stabilityScore = 0;
    let precisionScore = 0;
    let validFrames = 0;

    for (let i = 1; i < frameResults.length; i++) {
      const prev = frameResults[i - 1];
      const curr = frameResults[i];

      if (prev.landmarks.length > 0 && curr.landmarks.length > 0) {
        // Calculate stability (less movement = higher stability)
        const movement = this.calculateMovementBetweenFrames(prev.landmarks, curr.landmarks);
        stabilityScore += Math.max(0, 1 - movement * 5); // Scale movement

        // Calculate precision based on landmark consistency
        const precision = this.calculateLandmarkPrecision(curr.landmarks);
        precisionScore += precision;

        validFrames++;
      }
    }

    if (validFrames === 0) return 0;

    const avgStability = stabilityScore / validFrames;
    const avgPrecision = precisionScore / validFrames;
    
    // Combine scores with weights
    return Math.max(0, Math.min(100, (avgStability * 0.6 + avgPrecision * 0.4) * 100));
  }

  private countHandMovements(frameResults: HandResults[]): number {
    let movements = 0;
    const movementThreshold = 0.02; // Threshold for significant movement

    for (let i = 1; i < frameResults.length; i++) {
      const prev = frameResults[i - 1];
      const curr = frameResults[i];

      if (prev.landmarks.length > 0 && curr.landmarks.length > 0) {
        const movement = this.calculateMovementBetweenFrames(prev.landmarks, curr.landmarks);
        if (movement > movementThreshold) {
          movements++;
        }
      }
    }

    return movements;
  }

  private calculateAverageSpeed(frameResults: HandResults[]): number {
    let totalSpeed = 0;
    let validFrames = 0;

    for (let i = 1; i < frameResults.length; i++) {
      const prev = frameResults[i - 1];
      const curr = frameResults[i];
      const timeDiff = curr.timestamp - prev.timestamp;

      if (prev.landmarks.length > 0 && curr.landmarks.length > 0 && timeDiff > 0) {
        const movement = this.calculateMovementBetweenFrames(prev.landmarks, curr.landmarks);
        const speed = movement / timeDiff;
        totalSpeed += speed;
        validFrames++;
      }
    }

    return validFrames > 0 ? totalSpeed / validFrames : 0;
  }

  private detectTremor(frameResults: HandResults[]): number {
    const movements: number[] = [];
    
    for (let i = 1; i < frameResults.length; i++) {
      const prev = frameResults[i - 1];
      const curr = frameResults[i];

      if (prev.landmarks.length > 0 && curr.landmarks.length > 0) {
        const movement = this.calculateMovementBetweenFrames(prev.landmarks, curr.landmarks);
        movements.push(movement);
      }
    }

    if (movements.length < 10) return 0;

    // Calculate variance to detect tremor-like patterns
    const mean = movements.reduce((a, b) => a + b) / movements.length;
    const variance = movements.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / movements.length;
    
    // Higher variance indicates more tremor
    return Math.min(variance * 1000, 100);
  }

  private calculateCoordinationScore(frameResults: HandResults[]): number {
    let coordinationScore = 0;
    let validFrames = 0;

    for (const frame of frameResults) {
      if (frame.landmarks.length >= 2) {
        // Analyze coordination between both hands
        const leftHand = frame.landmarks[0];
        const rightHand = frame.landmarks[1];
        
        // Calculate symmetry and coordination
        const symmetry = this.calculateHandSymmetry(leftHand, rightHand);
        coordinationScore += symmetry;
        validFrames++;
      }
    }

    return validFrames > 0 ? Math.max(0, Math.min(100, (coordinationScore / validFrames) * 100)) : 50;
  }

  private calculateLandmarkPrecision(landmarks: HandLandmark[][]): number {
    let precision = 0;
    
    for (const hand of landmarks) {
      // Check if landmarks form expected hand structure
      const fingerTips = [4, 8, 12, 16, 20]; // Landmark indices for fingertips
      let fingerPrecision = 0;
      
      for (const tipIndex of fingerTips) {
        if (hand[tipIndex]) {
          // Higher confidence (closer to 0 in z) indicates better detection
          fingerPrecision += Math.max(0, 1 - Math.abs(hand[tipIndex].z));
        }
      }
      
      precision += fingerPrecision / fingerTips.length;
    }
    
    return landmarks.length > 0 ? precision / landmarks.length : 0;
  }

  private calculateHandSymmetry(leftHand: HandLandmark[], rightHand: HandLandmark[]): number {
    if (!leftHand || !rightHand || leftHand.length !== rightHand.length) {
      return 0;
    }

    let symmetryScore = 0;
    
    for (let i = 0; i < leftHand.length; i++) {
      // Compare mirrored positions (flip x coordinate)
      const leftPoint = leftHand[i];
      const rightPoint = rightHand[i];
      
      const dx = Math.abs((1 - leftPoint.x) - rightPoint.x);
      const dy = Math.abs(leftPoint.y - rightPoint.y);
      
      const distance = Math.sqrt(dx * dx + dy * dy);
      symmetryScore += Math.max(0, 1 - distance * 2);
    }
    
    return symmetryScore / leftHand.length;
  }
}

export default HandAnalysisService; 