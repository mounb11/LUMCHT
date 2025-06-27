import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler, ChartData, ChartOptions } from 'chart.js';

ChartJS.register( CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler );

interface AnalysisPlotGenericProps {
  data: Array<{ timestamp: number; [key: string]: number }>;
  label: string;
  yKey: string;
  yLabel: string;
  color: string;
}

export default function AnalysisPlotGeneric({ data, label, yKey, yLabel, color }: AnalysisPlotGenericProps) {
  const chartData: ChartData<'line'> = {
    labels: data.map(d => d.timestamp.toFixed(2)),
    datasets: [
      {
        label,
        data: data.map(d => d[yKey]),
        fill: true,
        backgroundColor: color.replace('1)', '0.2)'),
        borderColor: color,
        pointRadius: 1,
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: label,
        color: '#E5E7EB',
        font: { size: 16 },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: (context) => `${yLabel}: ${context.formattedValue}`,
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Time (s)', color: '#9CA3AF' },
        ticks: { color: '#9CA3AF' },
        grid: { color: 'rgba(255,255,255,0.1)' },
      },
      y: {
        title: { display: true, text: yLabel, color: '#9CA3AF' },
        ticks: { color: '#9CA3AF' },
        grid: { color: 'rgba(255,255,255,0.1)' },
      },
    },
  };

  return <div className="h-64"><Line options={options} data={chartData} /></div>;
} 