#include <vector>

#include "../ThreadPool.h"

template <typename Func>
class Integration
{
    // Main engine of concurrency
    SG::ThreadPool Pool;
    // Function and domain of integration
    Func function;
    double l; double r;
    // Number of the points in the grid
    std::size_t N;
    
public:
    Integration(const Func& _function, double _l, double _r, std::size_t _N, std::size_t numthreads) :
        function{_function}, l{_l}, r{_r}, N{_N}, Pool{numthreads} {}

    double Evaluate(std::size_t numblocks)
    {
        std::vector<double> subintegrals(numblocks, 0.0);
        std::vector<std::size_t> blocks = SG::BlockManager(N, numblocks);

        Pool.StartWorkers();
        for (std::size_t b=0; b<numblocks; ++b)
        {
            // Push tasks to the queue
            Pool.PushTask([&]
            {
                double result = 0;
                auto x = [](std::size_t i) { return l + i*((r - l)/N); };
            }
            );
        }
    }
};

int main()
{
    auto function = [](double x){ return x*x; };
    double l = -1.0;
    double r = 1.0;
    std::size_t N = 1000000;
    std::size_t numthreads = 3; // To be engaged in the calculations themselves

    Integration Integrator{function, l, r, N, numthreads};


    return 0;
}