//===-- Support/Timer.h - Interval Timing Support ---------------*- C++ -*-===//
//
// This file defines three classes: Timer, TimeRegion, and TimerGroup.
//
// The Timer class is used to track the amount of time spent between invocations
// of it's startTimer()/stopTimer() methods.  Given appropriate OS support it
// can also keep track of the RSS of the program at various points.  By default,
// the Timer will print the amount of time it has captured to standard error
// when the laster timer is destroyed, otherwise it is printed when it's
// TimerGroup is destroyed.  Timer's do not print their information if they are
// never started.
//
// The TimeRegion class is used as a helper class to call the startTimer() and
// stopTimer() methods of the Timer class.  When the object is constructed, it
// starts the timer specified as it's argument.  When it is destroyed, it stops
// the relevant timer.  This makes it easy to time a region of code.
//
// The TimerGroup class is used to group together related timers into a single
// report that is printed when the TimerGroup is destroyed.  It is illegal to
// destroy a TimerGroup object before all of the Timers in it are gone.  A
// TimerGroup can be specified for a newly created timer in its constructor.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TIMER_H
#define SUPPORT_TIMER_H

#include <string>
#include <vector>

class TimerGroup;

class Timer {
  double Elapsed;        // Wall clock time elapsed in seconds
  double UserTime;       // User time elapsed
  double SystemTime;     // System time elapsed
  unsigned long MaxRSS;  // Maximum resident set size (in bytes)
  unsigned long RSSTemp; // Temp for calculating maxrss
  std::string Name;      // The name of this time variable
  bool Started;          // Has this time variable ever been started?
  TimerGroup *TG;        // The TimerGroup this Timer is in.
public:
  Timer(const std::string &N);
  Timer(const std::string &N, TimerGroup &tg);
  Timer(const Timer &T);
  ~Timer();

  double getProcessTime() const { return UserTime+SystemTime; }
  double getWallTime() const { return Elapsed; }
  unsigned long getMaxRSS() const { return MaxRSS; }
  std::string   getName() const { return Name; }

  const Timer &operator=(const Timer &T) {
    Elapsed = T.Elapsed;
    UserTime = T.UserTime;
    SystemTime = T.SystemTime;
    MaxRSS = T.MaxRSS;
    RSSTemp = T.RSSTemp;
    Name = T.Name;
    Started = T.Started;
    assert (TG == T.TG && "Can only assign timers in the same TimerGroup!");
    return *this;
  }

  // operator< - Allow sorting...
  bool operator<(const Timer &T) const {
    // Primary sort key is User+System time
    if (UserTime+SystemTime < T.UserTime+T.SystemTime)
      return true;
    if (UserTime+SystemTime > T.UserTime+T.SystemTime)
      return false;
    
    // Secondary sort key is Wall Time
    return Elapsed < T.Elapsed;
  }
  bool operator>(const Timer &T) const { return T.operator<(*this); }
  
  /// startTimer - Start the timer running.  Time between calls to
  /// startTimer/stopTimer is counted by the Timer class.  Note that these calls
  /// must be correctly paired.
  ///
  void startTimer();

  /// stopTimer - Stop the timer.
  ///
  void stopTimer();

  /// print - Print the current timer to standard error, and reset the "Started"
  /// flag.
  void print(const Timer &Total);

private:
  friend class TimerGroup;

  // Copy ctor, initialize with no TG member.
  Timer(bool, const Timer &T);

  /// sum - Add the time accumulated in the specified timer into this timer.
  ///
  void sum(const Timer &T);
};


class TimeRegion {
  Timer &T;
  TimeRegion(const TimeRegion &); // DO NOT IMPLEMENT
public:
  TimeRegion(Timer &t) : T(t) {
    T.startTimer();
  }
  ~TimeRegion() {
    T.stopTimer();
  }
};

class TimerGroup {
  std::string Name;
  unsigned NumTimers;
  std::vector<Timer> TimersToPrint;
public:
  TimerGroup(const std::string &name) : Name(name), NumTimers(0) {}
  ~TimerGroup() {
    assert(NumTimers == 0 &&
           "TimerGroup destroyed before all contained timers!");
  }

private:
  friend class Timer;
  void addTimer() { ++NumTimers; }
  void removeTimer();
  void addTimerToPrint(const Timer &T) {
    TimersToPrint.push_back(Timer(true, T));
  }
};

#endif
