"use client";

import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler, ChartData, ChartOptions } from 'chart.js';

ChartJS.register( CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler );

interface AnalysisPlotProps {
  pathDataLeft: Array<{ timestamp: number; pathLength: number }>;
  pathDataRight: Array<{ timestamp: number; pathLength: number }>;
}

export default function AnalysisPlot({ pathDataLeft, pathDataRight }: AnalysisPlotProps) {
  const labels = pathDataLeft.length > 0 
    ? pathDataLeft.map(d => d.timestamp.toFixed(1)) 
    : pathDataRight.map(d => d.timestamp.toFixed(1));

  const data: ChartData<'line'> = {
    labels,
    datasets: [
      {
        label: 'Left Hand Path Length',
        data: pathDataLeft.map(d => d.pathLength),
        fill: true,
        backgroundColor: 'rgba(59, 130, 246, 0.2)', // Blue
        borderColor: 'rgba(59, 130, 246, 1)',
        pointRadius: 2,
        tension: 0.3,
      },
      {
        label: 'Right Hand Path Length',
        data: pathDataRight.map(d => d.pathLength),
        fill: true,
        backgroundColor: 'rgba(239, 68, 68, 0.2)', // Red
        borderColor: 'rgba(239, 68, 68, 1)',
        pointRadius: 2,
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true, // Show legend to distinguish hands
        position: 'top',
        labels: {
          color: '#E5E7EB'
        }
      },
      title: {
        display: true,
        text: 'Cumulative Hand Path Length',
        color: '#E5E7EB',
        font: {
          size: 16,
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: (context) => `Path Length: ${context.formattedValue} units`,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time (seconds)',
          color: '#9CA3AF',
        },
        ticks: {
          color: '#9CA3AF',
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        }
      },
      y: {
        title: {
          display: true,
          text: 'Path Length (units)',
          color: '#9CA3AF',
        },
        ticks: {
          color: '#9CA3AF',
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        }
      },
    },
  };

  return <div className="h-64">{<Line options={options} data={data} />}</div>;
} 