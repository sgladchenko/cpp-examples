#ifndef SGTHREADPOOL_H
#define SGTHREADPOOL_H

#include <queue>
#include <thread>
#include <mutex>
#include <functional>
#include <exception>

namespace SG
{
    // Basic object of a thread-safe queue that is guarded with a mutex
    template <typename T>
    class LockQueue
    {
        std::queue<T> queue;
        mutable std::mutex mutex;
    
    public:
        // Exception that will be thrown each time the queue is empty and we have no 
        // active tasks
        struct NoObjectsException : std::exception {};

        // Makes an empty queue
        LockQueue() {}

        // Modification of the queue
        void pushback(const T& object);
        void pushback(T&& object);
        T popfront();
    };

    // The main type for the queue containing the lambda expressions of the slave threads
    using TaskQueue_t = LockQueue<std::function<void()>>;

    // The object that suits as a callable in the construction of a thread object
    class Worker
    {
        TaskQueue_t& TaskQueue;
        bool& Working;
    
    public:
        Worker(TaskQueue_t& _TaskQueue, bool& _Working) :
            TaskQueue{_TaskQueue},
            Working{_Working} {}

        void operator()()
        {
            while (Working)
            // This section is active only when the master thread guided to grab
            // available tasks
            {
                try
                {
                    auto task = TaskQueue.popfront();
                    task();
                }
                catch(const TaskQueue_t::NoObjectsException& e) {}
            }
            // When we are out of the previous section, we have to firstly ensure
            // that the queue is empty
            bool empty = false;
            while (!empty)
            {
                try
                {
                    auto task = TaskQueue.popfront();
                    task();
                }
                catch(const TaskQueue_t::NoObjectsException& e) { empty = true; }
            }
        }
    };

    // The main class of a thread pool. It is supposed to be handled in the master thread, pushing tasks
    // to the queue of tasks
    class ThreadPool
    {
        TaskQueue_t TaskQueue;
        std::size_t NumWorkers;
        std::vector<std::thread> Workers;
        bool Working = false;
    
    public:
        ThreadPool(std::size_t _NumWorkers) : NumWorkers{_NumWorkers}
        {
            // First, check whether it's possible to allocate the number of threads specified
            if (NumWorkers > std::thread::hardware_concurrency()) { throw; }
            else { Workers.resize(NumWorkers); }
        }

        // If workers haven't been initialized yet, let's start them
        void StartWorkers()
        {
            if (!Working)
            {
                Working = true;
                for (std::size_t t=0; t < NumWorkers; t++)
                {
                    Workers[t] = std::thread{ Worker{TaskQueue, Working} };
                }
            }
        }

        // Turn off the loops where the threads are waiting for the available tasks
        void AwaitWorkers()
        {
            Working = false;
            for (auto& worker: Workers) { worker.join(); }
        }

        // Push a task to the task queue
        template <typename Func>
        void PushTask(Func&& task) // task is a forwarding reference
        {
            TaskQueue.pushback(std::forward<Func>(task));
        }
    };
}

// Push a copy of an lvalue, giving by the reference
template <typename T>
void SG::LockQueue<T>::pushback(const T& object)
{
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(object);
}

// Push an rvalue itself just by moving it to the queue
template <typename T>
void SG::LockQueue<T>::pushback(T&& object)
{
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(std::move(object));
}

// Pop ab object from the queue; if there are no elements, then an
// exception is invoked, which would signalize to the scope of the call
// that there are no elements
template <typename T>
T SG::LockQueue<T>::popfront()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (queue.size() == 0) { throw NoObjectsException(); }
    else
    {
        T result = std::move(queue.front());
        queue.pop();
        return result;
    }
}

#endif