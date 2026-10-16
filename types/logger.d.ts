export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'silent';

export interface IChildLogger {
  debug(msg: string, meta?: any): void;
  info(msg: string, meta?: any): void;
  warn(msg: string, meta?: any): void;
  error(msg: string, meta?: any): void;
  child(subTag: string): IChildLogger;
}

export interface ILogger {
  debug(tag: string, msg?: any, meta?: any): void;
  info(tag: string, msg?: any, meta?: any): void;
  warn(tag: string, msg?: any, meta?: any): void;
  error(tag: string, msg?: any, meta?: any): void;
  child(tag: string): IChildLogger;
  setLevel(level: LogLevel | number): void;
  getLevel(): LogLevel;
}

export declare const LOG_LEVELS: Record<LogLevel, number>;
export declare const LEVEL_NAMES: Record<number, LogLevel>;
export declare class ChildLogger implements IChildLogger {
  constructor(parent: Logger, tag: string);
  tag: string;
  debug(msg: string, meta?: any): void;
  info(msg: string, meta?: any): void;
  warn(msg: string, meta?: any): void;
  error(msg: string, meta?: any): void;
  child(subTag: string): ChildLogger;
}
export declare class Logger implements ILogger {
  constructor(initialLevel?: LogLevel | number);
  setLevel(level: LogLevel | number): void;
  getLevel(): LogLevel;
  debug(tag: string, msg?: any, meta?: any): void;
  info(tag: string, msg?: any, meta?: any): void;
  warn(tag: string, msg?: any, meta?: any): void;
  error(tag: string, msg?: any, meta?: any): void;
  child(tag: string): ChildLogger;
}
export declare const logger: Logger;
