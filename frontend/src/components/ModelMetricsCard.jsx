import React, { useState, useEffect } from 'react'
import styles from '../styles/ModelMetricsCard.module.css'

const ModelMetricsCard = () => {
  const [metrics, setMetrics] = useState(null)
  const [registry, setRegistry] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('metrics')

  // Get API base URL from environment or default to localhost:5000
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  const fetchMetrics = async () => {
    try {
      const [metricsRes, registryRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/model-metrics`),
        fetch(`${API_BASE_URL}/api/model-registry`)
      ])
      
      const metricsData = await metricsRes.json()
      const registryData = await registryRes.json()
      
      if (metricsData.success) setMetrics(metricsData.data)
      if (registryData.success) setRegistry(registryData.data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className={styles.card}>
        <div className={styles.loading}>Loading model metrics...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={styles.card}>
        <div className={styles.error}>Error: {error}</div>
      </div>
    )
  }

  const accuracy = metrics?.overall_accuracy || 0
  const accuracyStatus = accuracy >= 75 ? 'high' : accuracy >= 60 ? 'medium' : 'low'

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <h3>🤖 MLOps Model Monitor</h3>
        <div className={styles.badge}>MLflow v2.10</div>
      </div>

      {/* Tab Navigation */}
      <div className={styles.tabs}>
        <button
          className={`${styles.tabBtn} ${activeTab === 'metrics' ? styles.active : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          📊 Metrics
        </button>
        <button
          className={`${styles.tabBtn} ${activeTab === 'registry' ? styles.active : ''}`}
          onClick={() => setActiveTab('registry')}
        >
          📋 Registry
        </button>
      </div>

      {/* Metrics Tab */}
      {activeTab === 'metrics' && metrics && (
        <div className={styles.content}>
          {/* Accuracy Overview */}
          <div className={styles.metricsGrid}>
            <div className={styles.metricBox}>
              <div className={styles.label}>Overall Accuracy</div>
              <div className={`${styles.value} ${styles[`accuracy-${accuracyStatus}`]}`}>
                {accuracy.toFixed(1)}%
              </div>
              <div className={styles.sublabel}>
                {metrics.correct_predictions}/{metrics.total_predictions}
              </div>
            </div>

            <div className={styles.metricBox}>
              <div className={styles.label}>Recent Accuracy</div>
              <div className={`${styles.value} ${styles[`accuracy-${accuracy >= 70 ? 'high' : 'medium'}`]}`}>
                {metrics.recent_accuracy.toFixed(1)}%
              </div>
              <div className={styles.sublabel}>
                Last {metrics.recent_count} predictions
              </div>
            </div>

            <div className={styles.metricBox}>
              <div className={styles.label}>Total Predictions</div>
              <div className={styles.value}>{metrics.total_predictions}</div>
              <div className={styles.sublabel}>
                {metrics.correct_predictions} correct
              </div>
            </div>
          </div>

          {/* Accuracy Trend */}
          <div className={styles.trendBox}>
            <div className={styles.label}>Recent Trend (20 predictions)</div>
            <div className={styles.trend}>
              {metrics.trend?.map((correct, idx) => (
                <div
                  key={idx}
                  className={`${styles.trendBar} ${correct ? styles.correct : styles.incorrect}`}
                  title={correct ? '✓ Correct' : '✗ Incorrect'}
                />
              ))}
            </div>
          </div>

          {/* Status Indicator */}
          <div className={styles.statusBox}>
            <div className={styles.statusDot} style={{
              backgroundColor: accuracy >= 75 ? '#10b981' : accuracy >= 60 ? '#f59e0b' : '#ef4444'
            }} />
            <span>
              {accuracy >= 75 && '✓ Model performing well'}
              {accuracy >= 60 && accuracy < 75 && '⚠ Monitor performance'}
              {accuracy < 60 && '⚠ Performance degraded - check data'}
            </span>
          </div>
        </div>
      )}

      {/* Registry Tab */}
      {activeTab === 'registry' && registry && (
        <div className={styles.content}>
          <div className={styles.registryBox}>
            <div className={styles.label}>Registered Models</div>
            <div className={styles.modelsList}>
              {registry.models?.length > 0 ? (
                registry.models.map((model, idx) => (
                  <div key={idx} className={styles.modelItem}>
                    <div className={styles.modelHeader}>
                      <span className={styles.modelName}>{model.name}</span>
                      <span className={styles.modelVersion}>{model.version}</span>
                      <span className={styles.modelStatus}>{model.status}</span>
                    </div>
                    <div className={styles.modelMetrics}>
                      <span>Accuracy: {(model.metrics.accuracy * 100).toFixed(1)}%</span>
                      <span>F1: {(model.metrics.f1_score || 0).toFixed(3)}</span>
                      <span>Precision: {(model.metrics.precision || 0).toFixed(3)}</span>
                    </div>
                    <div className={styles.modelTime}>
                      {new Date(model.timestamp).toLocaleDateString()}
                    </div>
                  </div>
                ))
              ) : (
                <div className={styles.noModels}>No models registered yet</div>
              )}
            </div>

            <div className={styles.mlflowInfo}>
              <div className={styles.label}>MLflow Info</div>
              <div className={styles.infoBox}>
                <div><strong>Total Runs:</strong> {registry.mlflow_runs}</div>
                <div><strong>UI:</strong> <code>mlflow ui --backend-store-uri file:./mlruns</code></div>
                <div className={styles.hint}>Run this command in your backend directory to view MLflow UI</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className={styles.footer}>
        <span className={styles.lastUpdate}>
          Last updated: {new Date().toLocaleTimeString()}
        </span>
        <button className={styles.refreshBtn} onClick={fetchMetrics}>
          🔄 Refresh
        </button>
      </div>
    </div>
  )
}

export default ModelMetricsCard
