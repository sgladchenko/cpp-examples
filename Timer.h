#ifndef SGTIMER_H
#define SGTIMER_H

#include <chrono>

namespace SG
{
    // A short class needed for quick evaluation of the elapsed time between tick and tock calls
    template<class timeunit = std::chrono::milliseconds, class timeclock = std::chrono::steady_clock>
    class Timer
    {
        using timepoint = typename timeclock::time_point;
        private:
            timepoint _start;
            timepoint _end;
        public:
            void tick() { _end = {}; _start = timeclock::now(); }
            void tock() { _end = timeclock::now(); }

            // The main function that returns the duration object
            timeunit duration() const
            {
                if (_start == timepoint{} or _end == timepoint{}) throw;
                return std::chrono::duration_cast<timeunit>(_end - _start);
            }
    };
}

#endif