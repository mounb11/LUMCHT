import React from 'react';

interface NumericSeries {
  label: string;
  data: Array<{ value: number }>;
}

function calcStats(series: number[]): { min: number, max: number, mean: number, std: number } {
  if (!series.length) return { min: 0, max: 0, mean: 0, std: 0 };
  const mean = series.reduce((a, b) => a + b, 0) / series.length;
  const std = Math.sqrt(series.reduce((a, b) => a + (b - mean) ** 2, 0) / series.length);
  return {
    min: Math.min(...series),
    max: Math.max(...series),
    mean,
    std
  };
}

export default function MetricsSummaryTable({ analysisResult }: { analysisResult: any }) {
  const metrics = [
    {
      label: 'Velocity',
      series: (analysisResult.velocitySeries || []).map((d: any) => d.velocity)
    },
    {
      label: 'Acceleration',
      series: (analysisResult.accelerationSeries || []).map((d: any) => d.acceleration)
    },
    {
      label: 'Jerk',
      series: (analysisResult.jerkSeries || []).map((d: any) => d.jerk)
    },
    {
      label: 'Dimensionless Jerk',
      series: (analysisResult.dimensionlessJerkSeries || []).map((d: any) => d.dimensionlessJerk)
    },
  ];

  return (
    <table className="w-full text-sm text-left border border-border rounded-lg overflow-hidden">
      <thead>
        <tr className="bg-gray-800 text-gray-300">
          <th className="p-2">Metric</th>
          <th className="p-2">Min</th>
          <th className="p-2">Max</th>
          <th className="p-2">Mean</th>
          <th className="p-2">Std</th>
        </tr>
      </thead>
      <tbody>
        {metrics.map(m => {
          const stats = calcStats(m.series);
          return (
            <tr key={m.label} className="border-t border-border">
              <td className="p-2 font-medium">{m.label}</td>
              <td className="p-2">{stats.min.toExponential(3)}</td>
              <td className="p-2">{stats.max.toExponential(3)}</td>
              <td className="p-2">{stats.mean.toExponential(3)}</td>
              <td className="p-2">{stats.std.toExponential(3)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
} 