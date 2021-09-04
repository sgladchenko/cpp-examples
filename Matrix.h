#ifndef SGMATRIX_H
#define SGMATRIX_H

#include <iostream>
#include <iomanip>
#include <complex>
#include <exception>
#include <cassert>

#include "immintrin.h"

// While compiling under C++11, we can't use all the capabilites of the constexpr keyword
#define __SG_CONSTEXPR_MATRIX__ /*constexpr*/

namespace SG
{
    // Some pretty nice function used for evaluation of norms.
    // Note that when we're operating over the complexes, we must achieve
    // positiveness and realness of the result
    template <typename T>
    inline T square(const T& number) { return number * number; }

    // And instantiate for the complexes
    template <>
    inline std::complex<double> square(const std::complex<double>& number) { return std::norm(number); }

    template <>
    inline std::complex<float> square(const std::complex<float>& number) { return std::norm(number); }

    template <>
    inline std::complex<long double> square(const std::complex<long double>& number) { return std::norm(number); }

    // Class providing key data members
    template <typename T, std::size_t N, std::size_t M>
    struct BaseMatrix
    {
        // The inner structure containing the elements of the matrix
        static constexpr std::size_t bufsize = N*M;
        T data[bufsize];

        static constexpr std::size_t rows = N; // Might be used in some contexts, where 
        static constexpr std::size_t cols = M; // we don't possess information about the type
        static constexpr std::size_t minimal = (N < M) ? N : M;

        __SG_CONSTEXPR_MATRIX__ BaseMatrix() {}
        __SG_CONSTEXPR_MATRIX__ BaseMatrix(const T& object);
        __SG_CONSTEXPR_MATRIX__ BaseMatrix(std::initializer_list<std::initializer_list<T>> initlist);

        // Non-const/const accessors (matrix indices)
        __SG_CONSTEXPR_MATRIX__ T& operator()(std::size_t i, std::size_t j) { return data[i*cols + j]; }
        __SG_CONSTEXPR_MATRIX__ const T& operator()(std::size_t i, std::size_t j) const { return data[i*cols + j]; }
    };

    // Simple trace, which is basically the sum of elements on the main diagonal
    template <typename T, std::size_t N, std::size_t M>
    __SG_CONSTEXPR_MATRIX__
    T tr(const BaseMatrix<T,N,M>& matrix)
    {
        T result = matrix(0,0);
        for (std::size_t i=1; i<BaseMatrix<T,N,M>::minimal; ++i) { result += matrix(i,i); }
        return result;
    }

    // Frobenius (squared) norm of the matrix
    template <typename T, std::size_t N, std::size_t M>
    __SG_CONSTEXPR_MATRIX__
    T norm(const BaseMatrix<T,N,M>& matrix)
    {
        T result = square(matrix.data[0]);
        for (std::size_t i=1; i<BaseMatrix<T,N,M>::bufsize; ++i) { result += square<T>(matrix.data[i]); }
        return result;
    }
    
    template <typename T, std::size_t N, std::size_t M>
    struct Matrix : public BaseMatrix<T,N,M>
    {
        __SG_CONSTEXPR_MATRIX__ Matrix() {}
        __SG_CONSTEXPR_MATRIX__ Matrix(const T& object) : BaseMatrix<T,N,M>(object) {}
        __SG_CONSTEXPR_MATRIX__ Matrix(std::initializer_list<std::initializer_list<T>> initlist) :
                                BaseMatrix<T,N,M>(initlist) {}

        // Basic algebraic operations on matrices
        __SG_CONSTEXPR_MATRIX__ Matrix<T,N,M> operator+(const Matrix<T,N,M>& rhs) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<T,N,M> operator-(const Matrix<T,N,M>& rhs) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<T,N,M> operator*(const T& number) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<T,N,M> operator/(const T& number) const;

        __SG_CONSTEXPR_MATRIX__ Matrix<T,N,1> col(std::size_t j) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<T,1,M> row(std::size_t i) const;
    };

    // And finally, two special cases of the vectors
    template <typename T, std::size_t N> using ColVector = Matrix<T,N,1>;
    template <typename T, std::size_t N> using RowVector = Matrix<T,1,N>;

    // Simple commutator of two matrices
    template <typename T, std::size_t N>
    __SG_CONSTEXPR_MATRIX__
    Matrix<T,N,N> commutator(const Matrix<T,N,N>& matrix1, const Matrix<T,N,N>& matrix2)
    {
        return matrix1*matrix2 - matrix2*matrix1;
    }

    // Simple transpose of a matrix
    template <typename T, std::size_t N, std::size_t M>
    __SG_CONSTEXPR_MATRIX__
    Matrix<T,M,N> transpose(const Matrix<T,N,M>& matrix)
    {
        Matrix<T,M,N> result;
        for (std::size_t i=0; i<M; ++i)
        {
            for (std::size_t j=0; j<N; ++j) { result(i,j) = matrix(j,i); }
        }
        return result;
    }

    // Down below there's a separate implementation of several optimized
    // matrix types (fixed size, what makes possible applying AVX2/SSE intrinsics)

    struct Matrix4x4d : public BaseMatrix<double,4,4>
    {
        Matrix4x4d() {}
        Matrix4x4d(double object) : BaseMatrix<double,4,4>(object) {}
        Matrix4x4d(std::initializer_list<std::initializer_list<double>> initlist) :
                   BaseMatrix<double,4,4>(initlist) {}

        // Arithmetic operations that are implemented via SSE/AVX2 
        Matrix4x4d operator+(const Matrix4x4d& rhs) const;
        Matrix4x4d operator-(const Matrix4x4d& rhs) const;
        Matrix4x4d operator*(double number) const;
        Matrix4x4d operator/(double number) const;
        // Row by row! Because they are aligned in this way
    };

    /*
    // More operations for these matrices

    Matrix4x4d commutator(const Matrix4x4d& matrix1, const Matrix4x4d& matrix2)
    {
        return matrix1*matrix2 - matrix2*matrix1;
    }
    */

    // These are two separate types which made first of all in a view of the ability
    // of optimizing operations of multiplications on the matrices 4x4

    struct ColVector4d : public BaseMatrix<double,4,1>
    {
        ColVector4d() {}
        ColVector4d(double object) : BaseMatrix<double,4,1>(object) {}
        ColVector4d(std::initializer_list<double> initlist)
        {
            assert((initlist.size() == 4)); // TODO: Add possible check in compile time via static_assert
            std::size_t i = 0;
            for (auto& each: initlist) { data[i] = each; i++; }
        }

        // Arithmetic operations that are implemented via SSE/AVX2 
        ColVector4d operator+(const ColVector4d& rhs) const;
        ColVector4d operator-(const ColVector4d& rhs) const;
        ColVector4d operator*(double number) const;
        ColVector4d operator/(double number) const;
    };

    struct RowVector4d : public BaseMatrix<double,1,4>
    {
        RowVector4d() {}
        RowVector4d(double object) : BaseMatrix<double,1,4>(object) {}
        RowVector4d(std::initializer_list<double> initlist)
        {
            assert((initlist.size() == 4)); // TODO: Add possible check in compile time via static_assert
            std::size_t i = 0;
            for (auto& each: initlist) { data[i] = each; i++; }
        }

        // Arithmetic operations that are implemented via SSE/AVX2 
        RowVector4d operator+(const RowVector4d& rhs) const;
        RowVector4d operator-(const RowVector4d& rhs) const;
        RowVector4d operator*(double number) const;
        RowVector4d operator/(double number) const;
    };
}

template <typename T, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::BaseMatrix<T,N,M>::BaseMatrix(const T& object)
{
    for (auto& each: data) { each = object; }
}

template <typename T, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::BaseMatrix<T,N,M>::BaseMatrix(std::initializer_list<std::initializer_list<T>> initlist)
{
    // Check the validity of the provided initializer list containing 
    // the initial matrix
    assert((initlist.size() == N)); // TODO: Add possible check in compile time via static_assert
    for (const auto& each: initlist) { assert((each.size() == M)); }

    // Fill the data
    std::size_t i = 0;
    for (const auto& line: initlist)
    {
        for (const auto& each: line) { data[i] = each; ++i; }
    }
}

template <typename T, std::size_t N, std::size_t M>
std::ostream& operator<<(std::ostream& os, const SG::BaseMatrix<T,N,M>& object)
// 'object' may refer to all SG::Matrix, SG::Matrix4x4d,....
// as they all inherited the interface of the BaseMatrix subobject 
{
    // User-defined with and precision which were set before the invocation of this func
    auto width = os.width();
    auto precision = os.precision();

    os << std::setw(0) << "[[ ";
    for (std::size_t i=0; i<N; ++i)
    {
        if (i > 0) { os << std::setw(0) << " [ ";}
        for (std::size_t j=0; j<M; ++j)
        {// object may refer to both SG::Matrix and SG::Matrix4x4d
            os << std::setw(width) << std::setprecision(precision) << object(i,j) << " ";
        }
        os << std::setw(0) << "]";
        if (i < object.rows-1) { std::cout << std::setw(0) << std::endl; }
        else { std::cout << std::setw(0) << "]"; }
    }
    return os;
}

// Basic arithmetic operations

template <typename T, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator+(const SG::Matrix<T,N,M>& rhs) const
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<T,N,M>::data[i] + rhs.data[i];
    }
    return result;
}

template <typename T, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator-(const SG::Matrix<T,N,M>& rhs) const
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<T,N,M>::data[i] - rhs.data[i];
    }
    return result;
}

template <typename T, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator*(const T& number) const
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<T,N,M>::data[i] * number;
    }
    return result;
}

template <typename T, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<T,N,M> SG::Matrix<T,N,M>::operator/(const T& number) const
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<T,N,M>::data[i] / number;
    }
    return result;
}

// Multiplication on a number on the right hand side
template <typename T, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<T,N,M> operator*(const T& number, const SG::Matrix<T,N,M>& object)
{
    SG::Matrix<T,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<T,N,M>::bufsize; ++i)
    {
        result.data[i] = number * object.data[i];
    }
    return result;
}

// The matrix multiplication; note that it requires three dimensions
template <typename T, std::size_t N, std::size_t P, std::size_t M>
__SG_CONSTEXPR_MATRIX__
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

// Vectorized arithmetic operations on Matrix4x4d

// Vectorized arithmetic operations of ColVector4d

// Vectorized arithmetic operations of RowVector4d

#endif


// TODOs:
// 1. Realise multiplication of a matrix on a vector
// 2. SSE/AVX for N=4, M=4 cases (but for what types?... seems, for doubles.)
// 3. Exponential function (and here we will see possible growth in performance)