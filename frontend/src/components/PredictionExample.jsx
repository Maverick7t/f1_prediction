import React, { useState } from 'react';
import { getQualifying, fetchAndPredict, fetchCustomPredictions } from '../api';

/**
 * Example component showing different ways to fetch predictions
 * with the new auto-qualifying feature
 */
export default function PredictionExample() {
    const [predictions, setPredictions] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Method 1: Auto-fetch qualifying + predict in one call
    const handleAutoFetchPredict = async () => {
        setLoading(true);
        setError(null);

        try {
            const raceMeta = {
                race_key: "2024__17__Sao_Paulo_Grand_Prix",
                race_year: 2024,
                event: "Sao Paulo Grand Prix",
                circuit: "Interlagos"
            };

            // This will automatically fetch qualifying and then predict
            const result = await fetchAndPredict(raceMeta);
            setPredictions(result);
            console.log("Predictions:", result);
        } catch (err) {
            setError(err.message);
            console.error("Prediction error:", err);
        } finally {
            setLoading(false);
        }
    };

    // Method 2: Manual control - fetch qualifying, then predict
    const handleManualFetchPredict = async () => {
        setLoading(true);
        setError(null);

        try {
            const raceMeta = {
                race_key: "2024__17__Sao_Paulo_Grand_Prix",
                race_year: 2024,
                event: "Sao Paulo Grand Prix",
                circuit: "Interlagos"
            };

            // Step 1: Fetch qualifying
            console.log("Fetching qualifying...");
            const qualifying = await getQualifying(raceMeta);
            console.log("Qualifying data:", qualifying);

            // Step 2: Call predict with qualifying
            console.log("Generating predictions...");
            const payload = { ...raceMeta, qualifying };
            const result = await fetchCustomPredictions(payload);

            setPredictions(result);
            console.log("Predictions:", result);
        } catch (err) {
            setError(err.message);
            console.error("Prediction error:", err);
        } finally {
            setLoading(false);
        }
    };

    // Method 3: Let backend auto-fetch (no qualifying in payload)
    const handleBackendAutoFetch = async () => {
        setLoading(true);
        setError(null);

        try {
            const raceMeta = {
                race_key: "2024__17__Sao_Paulo_Grand_Prix",
                race_year: 2024,
                event: "Sao Paulo Grand Prix",
                circuit: "Interlagos"
                // Note: No qualifying field - backend will auto-fetch
            };

            // Backend will automatically fetch qualifying if not provided
            const result = await fetchCustomPredictions(raceMeta);
            setPredictions(result);
            console.log("Predictions:", result);
        } catch (err) {
            setError(err.message);
            console.error("Prediction error:", err);
        } finally {
            setLoading(false);
        }
    };

    // Method 4: Fetch only qualifying (for display purposes)
    const handleFetchQualifyingOnly = async () => {
        setLoading(true);
        setError(null);

        try {
            const raceMeta = {
                race_year: 2024,
                circuit: "Interlagos"
            };

            const qualifying = await getQualifying(raceMeta);
            console.log("Qualifying data:", qualifying);

            // Display qualifying data
            alert(`Pole position: ${qualifying[0]?.driver} (${qualifying[0]?.team})`);
        } catch (err) {
            setError(err.message);
            console.error("Qualifying fetch error:", err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{
            padding: '20px',
            backgroundColor: '#1a1f2e',
            borderRadius: '10px',
            color: '#e2e8f0'
        }}>
            <h2 style={{ marginBottom: '20px' }}>Prediction API Examples</h2>

            <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '10px',
                marginBottom: '20px'
            }}>
                <button
                    onClick={handleAutoFetchPredict}
                    disabled={loading}
                    style={{
                        padding: '10px 20px',
                        backgroundColor: '#10b981',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        opacity: loading ? 0.6 : 1
                    }}
                >
                    Method 1: Auto-Fetch + Predict (One Call)
                </button>

                <button
                    onClick={handleManualFetchPredict}
                    disabled={loading}
                    style={{
                        padding: '10px 20px',
                        backgroundColor: '#3b82f6',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        opacity: loading ? 0.6 : 1
                    }}
                >
                    Method 2: Manual (Fetch Qualifying → Predict)
                </button>

                <button
                    onClick={handleBackendAutoFetch}
                    disabled={loading}
                    style={{
                        padding: '10px 20px',
                        backgroundColor: '#8b5cf6',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        opacity: loading ? 0.6 : 1
                    }}
                >
                    Method 3: Backend Auto-Fetch (No Qualifying)
                </button>

                <button
                    onClick={handleFetchQualifyingOnly}
                    disabled={loading}
                    style={{
                        padding: '10px 20px',
                        backgroundColor: '#f59e0b',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        opacity: loading ? 0.6 : 1
                    }}
                >
                    Method 4: Fetch Qualifying Only
                </button>
            </div>

            {loading && (
                <div style={{
                    padding: '15px',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    borderRadius: '5px',
                    marginBottom: '15px'
                }}>
                    Loading predictions...
                </div>
            )}

            {error && (
                <div style={{
                    padding: '15px',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '5px',
                    color: '#fca5a5',
                    marginBottom: '15px'
                }}>
                    <strong>Error:</strong> {error}
                </div>
            )}

            {predictions && (
                <div style={{
                    padding: '15px',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                    borderRadius: '5px',
                    marginBottom: '15px'
                }}>
                    <h3 style={{ marginBottom: '10px' }}>Results:</h3>
                    <div>
                        <strong>Winner:</strong> {predictions.winner_prediction?.driver}
                        ({predictions.winner_prediction?.percentage}%)
                    </div>
                    <div style={{ marginTop: '10px' }}>
                        <strong>Top 3:</strong>
                        <ol style={{ marginLeft: '20px', marginTop: '5px' }}>
                            {predictions.top3_prediction?.slice(0, 3).map((pred, idx) => (
                                <li key={idx}>
                                    {pred.driver} - {pred.team} ({pred.percentage}%)
                                </li>
                            ))}
                        </ol>
                    </div>
                </div>
            )}

            <div style={{
                padding: '15px',
                backgroundColor: 'rgba(100, 116, 139, 0.1)',
                border: '1px solid rgba(100, 116, 139, 0.3)',
                borderRadius: '5px',
                fontSize: '12px',
                lineHeight: '1.6'
            }}>
                <strong>How it works:</strong>
                <ul style={{ marginLeft: '20px', marginTop: '8px' }}>
                    <li><strong>Method 1:</strong> Frontend fetches qualifying, then calls predict</li>
                    <li><strong>Method 2:</strong> Same as Method 1, but with manual steps</li>
                    <li><strong>Method 3:</strong> Backend auto-fetches qualifying (simplest)</li>
                    <li><strong>Method 4:</strong> Fetch only qualifying data for display</li>
                </ul>
                <div style={{ marginTop: '10px' }}>
                    Check browser console for detailed logs.
                </div>
            </div>
        </div>
    );
}
