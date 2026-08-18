type IconProps = { size?: number };

export function MicIcon({ size = 20 }: IconProps) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none"><rect x="8" y="3" width="8" height="12" rx="4" stroke="currentColor" strokeWidth="1.8"/><path d="M5 11.5a7 7 0 0 0 14 0M12 18.5V22M9 22h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>;
}

export function KeyboardIcon({ size = 20 }: IconProps) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none"><rect x="3" y="6" width="18" height="12" rx="2.5" stroke="currentColor" strokeWidth="1.8"/><path d="M7 10h.01M10 10h.01M13 10h.01M16 10h.01M7 14h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>;
}

export function ArrowIcon({ size = 18 }: IconProps) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none"><path d="M5 12h13M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

export function StopIcon({ size = 18 }: IconProps) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>;
}

export function SparkIcon({ size = 18 }: IconProps) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none"><path d="m12 3 1.45 5.55L19 10l-5.55 1.45L12 17l-1.45-5.55L5 10l5.55-1.45L12 3ZM19 16l.55 2.45L22 19l-2.45.55L19 22l-.55-2.45L16 19l2.45-.55L19 16Z" fill="currentColor"/></svg>;
}
