#ifndef SGMATRIX_H
#define SGMATRIX_H

#include <iostream>
#include <iomanip>

namespace SG
{

    template <typename T, std::size_t N, std::size_t M>
    class Matrix
    {
        static constexpr std::size_t bufsize = N*M;
        T data[bufsize];

    public:
        static constexpr std::size_t rows = N;
        static constexpr std::size_t cols = M;

        constexpr Matrix() : data {} {}
        constexpr Matrix(const T& object);
        constexpr Matrix(std::initializer_list<std::initializer_list<T>> initlist);

        // Non-const/const accessors (matrix indices)
        constexpr T& operator()(std::size_t i, std::size_t j) { return data[i*cols + j]; }
        constexpr const T& operator()(std::size_t i, std::size_t j) const { return data[i*cols + j]; }

        // Serialized index access
        constexpr T& operator[](std::size_t i) { return data[i]; }
        constexpr const T& operator[](std::size_t i) const { return data[i]; }

        // Basic algebraic operations over matrices
        constexpr Matrix<T,N,M> operator+(const Matrix<T,N,M>& rhs) const noexcept;
        constexpr Matrix<T,N,M> operator-(const Matrix<T,N,M>& rhs) const noexcept;
        constexpr Matrix<T,N,M> operator*(const T& number) const noexcept;
        constexpr Matrix<T,N,M> operator/(const T& number) const noexcept;
        friend constexpr Matrix<T,N,M> operator*(const T& number, const Matrix<T,N,M>& object)
        {
            Matrix<T,N,M> result {};
            for (std::size_t i=0; i<Matrix<T,N,M>::bufsize; ++i)
            {
                result.data[i] = number * object.data[i];
            }
            return result;
        }

    };

    /* --- TO-DO's --- */
    // (*) More operations on matrices (commutator, det, trace, transpose...)
    // (*) Realise separately the same type but for the matrcies 4x4 with AVX/SSE operations involved
}

template <typename T, std::size_t N, std::size_t M>
constexpr
SG::Matrix<T,N,M>::Matrix(const T& object) : data {}
{
    for (auto& each: data) { each = object; }
}

template <typename T, std::size_t N, std::size_t M>
constexpr
SG::Matrix<T,N,M>::Matrix(std::initializer_list<std::initializer_list<T>> initlist) : data {}
{
    // Check the validity of the provided initializer list containing 
    // the initial matrix
    if (initlist.size() != N) { throw; }
    for (const auto& each: initlist) { if (each.size() != M) { throw; } }

    // Fill the data
    std::size_t i = 0;
    for (const auto& line: initlist)
    {
        for (const auto& each: line) { data[i] = each; ++i; }
    }
}

template <typename T, std::size_t N, std::size_t M>
std::ostream& operator<<(std::ostream& os, const SG::Matrix<T,N,M>& object)
{
    // User-defined with and precision which were set before the invocation of this func
    auto width = os.width();
    auto precision = os.precision();

    os << std::setw(0) << "[[ ";
    for (std::size_t i=0; i<N; ++i)
    {
        if (i > 0) { os << std::setw(0) << " [ ";}
        for (std::size_t j=0; j<M; ++j)
        {
            os << std::setw(width) << std::setprecision(precision) << object(i,j) << " ";
        }
        os << std::setw(0) << "]";
        if (i < object.rows-1) { std::cout << std::setw(0) << std::endl; }
        else { std::cout << std::setw(0) << "]"; }
    }
    return os;
}

// Basic algebraic operations

template <typename T, std::size_t N, std::size_t M>
constexpr
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator+(const SG::Matrix<T,N,M>& rhs) const noexcept
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = data[i] + rhs.data[i];
    }
    return result;
}

template <typename T, std::size_t N, std::size_t M>
constexpr
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator-(const SG::Matrix<T,N,M>& rhs) const noexcept
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = data[i] - rhs.data[i];
    }
    return result;
}

template <typename T, std::size_t N, std::size_t M>
constexpr
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator*(const T& number) const noexcept
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = data[i] * number;
    }
    return result;
}

template <typename T, std::size_t N, std::size_t M>
constexpr
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator/(const T& number) const noexcept
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = data[i] / number;
    }
    return result;
}

// The matrix multiplication; note that it requires three dimensions
template <typename T, std::size_t N, std::size_t P, std::size_t M>
constexpr
SG::Matrix<T,N,M> operator*(const SG::Matrix<T,N,P>& object1, const SG::Matrix<T,P,M>& object2) 
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<N; ++i)
    {
        for (std::size_t j=0; j<M; ++j)
        {
            T tmp = object1(i,0) * object2(0,j);
            for (std::size_t k=1; k<P; ++k) { tmp += object1(i,k) * object2(k,j); }
            result(i,j) = tmp;
        }
    }
    return result;
}

#endif