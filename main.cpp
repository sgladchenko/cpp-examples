#include <iostream>

template <typename T, std::size_t N>
struct Base
{
    T data[N];
};

template <typename T, std::size_t N>
struct Derived : Base<T,N>
{
    T operator[](std::size_t i) { return data[i]; } 
};

class Derived<int,5>;

int main()
{
    std::cout << "Hello!\n";
    return 0;
}