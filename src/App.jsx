import React from 'react';
import { HeroSection } from './components/sections/HeroSection';
import { EmotionalCheckIn } from './components/sections/EmotionalCheckIn';
import { PreferencesSection } from './components/sections/PreferencesSection';
import { AnalysisSection } from './components/sections/AnalysisSection';
import { RecommendationDeck } from './components/sections/RecommendationDeck';
import { BookingSummary } from './components/sections/BookingSummary';
import { CelebrationSection } from './components/sections/CelebrationSection';
import { AmbientCursor } from './components/ui/AmbientCursor';
import { AppProviders } from './providers/AppProviders';
import { SerenityLayout } from './layouts/SerenityLayout';
import { CinematicJourney } from './components/interactive/CinematicJourney';
import './styles/globals.css';

function App() {
  return (
    <div className="font-['DM_Sans'] text-[#0B1728]">
      <AmbientCursor />
      <SerenityLayout>
        <AppProviders>
          <HeroSection />
          <EmotionalCheckIn />
          <PreferencesSection />
          <AnalysisSection />
          <RecommendationDeck />
          <BookingSummary />
          <CelebrationSection />
        </AppProviders>
        <CinematicJourney />
      </SerenityLayout>
    </div>
  );
}

export default App;