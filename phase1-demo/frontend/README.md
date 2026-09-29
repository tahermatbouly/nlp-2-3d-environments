# Frontend Redesign

The frontend has been redesigned to be clean and elegant with a dark theme inspired by Apple's design principles.

## Features

- Dark color scheme with subtle gradients and blur effects
- Initial view showing only the app name and two action buttons (Text Input and Voice Input)
- Smooth transitions and animations
- Retained all original functionality:
  - Text-based chat with the AI
  - Voice input with real-time transcription
  - Visualization of bubble constraint graphs
  - 2D floor plan preview
  - State inspection and reset capabilities

## Design Details

- Color palette: Blacks and dark grays with Apple blue (#0a84ff) as primary accent
- Typography: Inter font for a modern, readable appearance
- Layout: Responsive grid that adapts to mobile and desktop
- Interactive elements: Hover effects, focus states, and tactile feedback
- Visualizations: Maintained vis.js for bubble diagrams with adjusted theming

## Implementation

The redesign involved:
1. Replacing the entire index.html with a new structure
2. Updating all CSS variables for dark theme
3. Adding view management (initial view vs chat view)
4. Preserving all original JavaScript functionality with minor DOM adjustments
5. Enhancing visual feedback for user interactions

## Usage

1. Application opens to initial view with "FloorPlan AI" title and two buttons
2. Click "Enter Text" to show chat view focused on text input
3. Click "Speak to Model" to show chat view and immediately start voice input
4. Chat view provides full interface for interaction and visualization
5. Reset button clears all data and returns to initial state

## Compatibility

- Works with existing FastAPI backend endpoints unchanged
- Maintains compatibility with all original features
- Responsive design supports mobile and desktop browsers