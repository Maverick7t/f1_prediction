import React, { useState, useEffect } from 'react'
import Header from './Header'
import DriverCard from './DriverCard'
import RaceInfoCard from './RaceInfoCard'
import WinnerPredictionCard from './WinnerPredictionCard'
import RaceHistoryCard from './RaceHistoryCard'
import MatchupCard from './MatchupCard'
import EnhancedCircuitMapCard from './EnhancedCircuitMapCard'
import StandingsView from './StandingsView'
import ConstructorStandingsView from './ConstructorStandingsView'
import ModelMetricsCard from './ModelMetricsCard'
import { fetchSaoPauloPredictions, transformPredictionsToDriverData, fetchNextRace, fetchCurrentDrivers, fetchConstructorStandings, fetchRaceHistory } from '../api'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('current')
  const [driverData, setDriverData] = useState([])
  const [winnerPrediction, setWinnerPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [nextRace, setNextRace] = useState(null)
  const [openF1Drivers, setOpenF1Drivers] = useState([])

  const [constructorStandings, setConstructorStandings] = useState([])
  const [raceHistory, setRaceHistory] = useState([])

  // Fetch predictions on component mount
  useEffect(() => {
    async function loadPredictions() {
      try {
        setLoading(true)

        // Fetch next race from OpenF1
        try {
          const raceInfo = await fetchNextRace()
          setNextRace(raceInfo)
          console.log('✓ Next race loaded:', raceInfo.raceName)
        } catch (err) {
          console.error('❌ Failed to load next race info:', err)
          setNextRace({ error: err.message })
        }

        // Fetch current drivers from OpenF1
        try {
          const drivers = await fetchCurrentDrivers()
          setOpenF1Drivers(drivers)
          console.log(`✓ Loaded ${drivers.length} drivers from OpenF1`)
        } catch (err) {
          console.error('❌ Failed to load drivers info:', err)
        }

        // Fetch constructor standings
        try {
          const constructors = await fetchConstructorStandings()
          setConstructorStandings(constructors)
          console.log('✓ Constructor standings loaded')
        } catch (err) {
          console.error('❌ Failed to load constructor standings:', err)
        }

        // Fetch race history
        try {
          const raceHistoryData = await fetchRaceHistory()
          setRaceHistory(raceHistoryData)
          console.log('✓ Race history loaded:', raceHistoryData.length, 'races')
        } catch (err) {
          console.error('❌ Failed to load race history:', err)
          setRaceHistory([])
        }

        // Fetch São Paulo predictions
        try {
          const predictions = await fetchSaoPauloPredictions()

          // Transform predictions to driver data
          const drivers = transformPredictionsToDriverData(predictions)
          setDriverData(drivers)

          // Set winner prediction
          setWinnerPrediction({
            driver: predictions.winner_prediction.driver,
            team: predictions.winner_prediction.team,
            percentage: predictions.winner_prediction.percentage
          })

          setError(null)
          console.log('✓ All predictions loaded from backend')
        } catch (err) {
          console.error('❌ Failed to load predictions:', err)
          setError('Failed to load predictions from API. Please ensure the backend is running.')
          setDriverData([])
          setWinnerPrediction(null)
        }
      } catch (outerErr) {
        console.error('❌ Unexpected error in loadPredictions:', outerErr)
        setError('An unexpected error occurred while loading data.')
      } finally {
        setLoading(false)
      }
    }

    loadPredictions()
  }, [])

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      minHeight: '100vh',
      padding: '24px 32px',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      color: '#e2e8f0'
    }}>
      {/* Header Component */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Loading State */}
      {loading && (
        <div style={{
          textAlign: 'center',
          padding: '60px',
          color: '#94a3b8',
          fontSize: '14px'
        }}>
          <div style={{
            fontSize: '18px',
            fontWeight: '700',
            marginBottom: '12px'
          }}>Loading Predictions...</div>
          <div>Fetching data from ML model...</div>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div style={{
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '10px',
          padding: '20px',
          margin: '20px 0',
          color: '#fca5a5',
          fontSize: '13px'
        }}>
          <div style={{ fontWeight: '700', marginBottom: '8px' }}>⚠️ {error}</div>
          <div style={{ fontSize: '12px', opacity: 0.8 }}>Showing fallback data. Make sure the API server is running on {API_BASE_URL}</div>
        </div>
      )}

      {/* Current Race Tab Content */}
      {!loading && activeTab === 'current' && (
        driverData.length === 0 && error ? (
          <div style={{
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '10px',
            padding: '40px',
            textAlign: 'center',
            maxWidth: '600px'
          }}>
            <div style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#fca5a5',
              marginBottom: '12px'
            }}>
              Unable to Load Data
            </div>
            <div style={{
              fontSize: '13px',
              color: '#cbd5e1',
              marginBottom: '20px'
            }}>
              {error}
            </div>
            <div style={{
              fontSize: '12px',
              color: '#94a3b8'
            }}>
              Please check that:
              <ul style={{
                listStyle: 'none',
                padding: 0,
                marginTop: '12px',
                textAlign: 'left',
                maxWidth: '400px',
                margin: '12px auto 0'
              }}>
                <li>✓ Backend server is running on {API_BASE_URL}</li>
                <li>✓ VITE_API_URL environment variable is set correctly</li>
                <li>✓ OpenF1 API is accessible</li>
              </ul>
            </div>
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1.8fr',
            gap: '28px',
            maxWidth: '1600px'
          }}>
            {/* Left Column */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '20px'
            }}>
              <RaceInfoCard
                raceName={nextRace?.raceName || "Abu Dhabi Grand Prix"}
                dates={nextRace?.dateStart ? nextRace.dateStart.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }).toUpperCase() : "DECEMBER 2024"}
                time={nextRace?.dateStart ? nextRace.dateStart.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' }) : "TBD"}
                track={nextRace?.circuitName || "Yas Marina Circuit"}
                country={nextRace?.country || "United Arab Emirates"}
                circuitImage={nextRace?.circuitImage || "https://media.formula1.com/image/upload/f_auto,c_limit,w_1440,q_auto/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Abu_Dhabi_Circuit"}
              />
              <WinnerPredictionCard
                percentage={winnerPrediction?.percentage || 72}
                driverName={winnerPrediction?.driver || 'NOR'}
                teamColor="#ea580c"
                headshotUrl={openF1Drivers.find(d => d.nameAcronym === winnerPrediction?.driver)?.headshotUrl}
                fullName={openF1Drivers.find(d => d.nameAcronym === winnerPrediction?.driver)?.fullName}
                confidence={winnerPrediction?.confidence || 'HIGH'}
                confidenceColor={winnerPrediction?.confidence_color || '#f59e0b'}
              />

            </div>

            {/* Right Column */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '20px'
            }}>
              {/* Driver Predictions Grid - 2x3 */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '14px'
              }}>
                {driverData.map((driver, idx) => {
                  // Find matching driver from OpenF1
                  const openF1Driver = openF1Drivers.find(
                    d => d.nameAcronym === driver.name ||
                      d.teamName === driver.team
                  );

                  return (
                    <DriverCard
                      key={idx}
                      name={driver.name}
                      team={driver.team}
                      percentage={driver.percentage}
                      teamColor={openF1Driver?.teamColour || driver.teamColor}
                      position={driver.position}
                      points={driver.points}
                      headshotUrl={openF1Driver?.headshotUrl}
                      fullName={openF1Driver?.fullName}
                      confidence={driver.confidence || 'MEDIUM'}
                      confidenceColor={driver.confidenceColor || '#f59e0b'}
                    />
                  );
                })}
              </div>

              {/* Race History Table */}
              <RaceHistoryCard raceHistory={raceHistory} />
            </div>
          </div>
        )
      )}

      {/* Driver Standings Tab */}
      {!loading && activeTab === 'standings' && (
        <div style={{
          maxWidth: '1000px'
        }}>
          <StandingsView />
        </div>
      )}

      {/* Constructor Standings Tab */}
      {!loading && activeTab === 'constructor' && (
        <div style={{
          maxWidth: '1000px'
        }}>
          <ConstructorStandingsView />
        </div>
      )}

      {/* Circuit Map Tab */}
      {!loading && activeTab === 'circuit' && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '28px',
          maxWidth: '1400px'
        }}>
          <EnhancedCircuitMapCard />
        </div>
      )}

      {/* Matchup Tab */}
      {!loading && activeTab === 'matchup' && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '28px',
          maxWidth: '1200px'
        }}>
          <MatchupCard />
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '20px'
          }}>
            <div style={{
              backgroundColor: '#1a1f2e',
              border: '1px solid #334155',
              borderRadius: '10px',
              padding: '18px'
            }}>
              <div style={{
                fontSize: '11px',
                fontWeight: '700',
                letterSpacing: '1px',
                color: '#94a3b8',
                textTransform: 'uppercase',
                marginBottom: '12px'
              }}>
                HOW THIS WORKS
              </div>
              <div style={{
                fontSize: '11px',
                color: '#cbd5e1',
                lineHeight: '1.8'
              }}>
                <p>Select any two drivers to compare their predicted performance in the upcoming race.</p>
                <p style={{ marginTop: '12px' }}>The model analyzes:</p>
                <ul style={{ marginLeft: '16px', marginTop: '8px', listStyle: 'none' }}>
                  <li>📊 Recent form & race results</li>
                  <li>🏁 Circuit-specific performance</li>
                  <li>⛅ Weather predictions</li>
                  <li>🔧 Vehicle setup & reliability</li>
                  <li>👥 Head-to-head history</li>
                </ul>
              </div>
            </div>
            <div style={{
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: '10px',
              padding: '16px'
            }}>
              <div style={{
                fontSize: '11px',
                fontWeight: '700',
                color: '#10b981',
                marginBottom: '8px'
              }}>
                💡 PREDICTION TIP
              </div>
              <div style={{
                fontSize: '10px',
                color: '#cbd5e1',
                lineHeight: '1.6'
              }}>
                Check for weather updates 24 hours before the race - wet conditions can dramatically shift these probabilities!
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MLOps Model Monitor Tab */}
      {!loading && activeTab === 'mlops' && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '28px',
          maxWidth: '1200px'
        }}>
          <ModelMetricsCard />
        </div>
      )}

    </div>
  )
}
