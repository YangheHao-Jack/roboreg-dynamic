// roslib ships no types and @types/roslib isn't installed.
// Declare the whole module as any so `import * as ROSLIB from 'roslib'`
// type-checks (ROSLIB.Ros / ROSLIB.Topic become any).
declare module 'roslib';
