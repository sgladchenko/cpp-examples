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
    template <typename Type>
    inline Type square(const Type& number) { return number * number; }

    // And instantiate for the complexes
    template <>
    inline std::complex<double>
    square(const std::complex<double>& number) { return std::norm(number); }

    template <>
    inline std::complex<float>
    square(const std::complex<float>& number) { return std::norm(number); }

    template <>
    inline std::complex<long double>
    square(const std::complex<long double>& number) { return std::norm(number); }

    // Class providing key data members and function members for all the matrix types
    template <typename Type, std::size_t N, std::size_t M, std::size_t alignment = 0>
    struct BaseMatrix
    {
        // The inner structure containing the elements of the matrix
        static constexpr std::size_t bufsize = N*M;
        alignas(alignment) Type data[bufsize];

        static constexpr std::size_t rows = N; // Might be used in some contexts, where 
        static constexpr std::size_t cols = M; // we don't possess information about the type
        static constexpr std::size_t minimal = (N < M) ? N : M;

        __SG_CONSTEXPR_MATRIX__ BaseMatrix() {}
        __SG_CONSTEXPR_MATRIX__ BaseMatrix(const Type& object);
        __SG_CONSTEXPR_MATRIX__ BaseMatrix(std::initializer_list<std::initializer_list<Type>> initlist);

        // Non-const/const accessors (matrix indices)
        __SG_CONSTEXPR_MATRIX__ Type& operator()(std::size_t i, std::size_t j) { return data[i*cols + j]; }
        __SG_CONSTEXPR_MATRIX__ const Type& operator()(std::size_t i, std::size_t j) const { return data[i*cols + j]; }
    };

    // Simple trace, which is basically the sum of elements on the main diagonal
    template <typename Type, std::size_t N, std::size_t M, std::size_t alignment>
    __SG_CONSTEXPR_MATRIX__
    Type tr(const BaseMatrix<Type,N,M,alignment>& matrix)
    {
        Type result = matrix(0,0);
        for (std::size_t i=1; i<BaseMatrix<Type,N,M>::minimal; ++i) { result += matrix(i,i); }
        return result;
    }

    // Frobenius (squared) norm of the matrix
    template <typename Type, std::size_t N, std::size_t M, std::size_t alignment>
    __SG_CONSTEXPR_MATRIX__
    Type norm(const BaseMatrix<Type,N,M,alignment>& matrix)
    {
        Type result = square(matrix.data[0]);
        for (std::size_t i=1; i<BaseMatrix<Type,N,M>::bufsize; ++i) { result += square<Type>(matrix.data[i]); }
        return result;
    }
    
    template <typename Type, std::size_t N, std::size_t M>
    struct Matrix : public BaseMatrix<Type,N,M>
    {
        __SG_CONSTEXPR_MATRIX__ Matrix() {}
        __SG_CONSTEXPR_MATRIX__ Matrix(const Type& object) : BaseMatrix<Type,N,M>(object) {}
        __SG_CONSTEXPR_MATRIX__ Matrix(std::initializer_list<std::initializer_list<Type>> initlist) :
                                BaseMatrix<Type,N,M>(initlist) {}

        // Basic algebraic operations on matrices
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M> operator+(const Matrix<Type,N,M>& rhs) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M> operator-(const Matrix<Type,N,M>& rhs) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M> operator*(const Type& number) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M> operator/(const Type& number) const;

        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M>& operator+=(const Matrix<Type,N,M>& rhs) { *this = *this + rhs; return *this; }
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M>& operator-=(const Matrix<Type,N,M>& rhs) { *this = *this - rhs; return *this; }
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M>& operator*=(const Type& object) { *this = *this * object; return *this; }
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,M>& operator/=(const Type& object) { *this = *this / object; return *this; }

        // Methods that return separate columns and rows of the matrix
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,N,1> col(std::size_t j) const;
        __SG_CONSTEXPR_MATRIX__ Matrix<Type,1,M> row(std::size_t i) const;

        __SG_CONSTEXPR_MATRIX__ Matrix<Type,M,N> T() const { return transpose(*this); }
    };

    // And finally, two special cases of the vectors
    template <typename Type, std::size_t N> using ColVector = Matrix<Type,N,1>;
    template <typename Type, std::size_t N> using RowVector = Matrix<Type,1,N>;

    // Simple commutator of two matrices
    template <typename Type, std::size_t N>
    __SG_CONSTEXPR_MATRIX__ inline
    Matrix<Type,N,N> commutator(const Matrix<Type,N,N>& matrix1, const Matrix<Type,N,N>& matrix2)
    {
        return matrix1*matrix2 - matrix2*matrix1;
    }

    // Simple transpose of a matrix
    template <typename Type, std::size_t N, std::size_t M>
    __SG_CONSTEXPR_MATRIX__
    Matrix<Type,M,N> transpose(const Matrix<Type,N,M>& matrix)
    {
        Matrix<Type,M,N> result {};
        for (std::size_t i=0; i<M; ++i)
        {
            for (std::size_t j=0; j<N; ++j) { result(i,j) = matrix(j,i); }
        }
        return result;
    }

    // Down below there's a separate implementation of several optimized
    // matrix types (fixed size, what makes possible applying AVX2/SSE intrinsics)

    struct Matrix4x4d : BaseMatrix<double,4,4,256>
    {
        Matrix4x4d() {}
        Matrix4x4d(double object) : BaseMatrix<double,4,4,256>(object) {}
        Matrix4x4d(std::initializer_list<std::initializer_list<double>> initlist) :
                   BaseMatrix<double,4,4,256>(initlist) {}

        // Arithmetic operations that are implemented via SSE/AVX2 
        // Row by row! Because they are aligned in this way
        Matrix4x4d operator+(const Matrix4x4d& rhs) const;
        Matrix4x4d operator-(const Matrix4x4d& rhs) const;
        Matrix4x4d operator*(double number) const;
        Matrix4x4d operator/(double number) const;

        Matrix4x4d& operator+=(const Matrix4x4d& rhs) { *this = *this + rhs; return *this; }
        Matrix4x4d& operator-=(const Matrix4x4d& rhs) { *this = *this - rhs; return *this; }
        Matrix4x4d& operator*=(double number) { *this = *this * number; return *this; }
        Matrix4x4d& operator/=(double number) { *this = *this / number; return *this; }

        Matrix4x4d T() const;
    };

    // More operations for these matrices

    inline Matrix4x4d
    commutator(const Matrix4x4d& matrix1, const Matrix4x4d& matrix2);

    Matrix4x4d transpose(const Matrix4x4d& matrix);

    // These are two separate types which made first of all in a view of the ability
    // of optimizing operations of multiplications on the matrices 4x4

    struct ColVector4d : public BaseMatrix<double,4,1,256>
    {
        ColVector4d() {}
        ColVector4d(double object) : BaseMatrix<double,4,1,256>(object) {}
        ColVector4d(std::initializer_list<double> initlist)
        {
            assert((initlist.size() == 4)); // TODO: Add possible check in compile time via static_assert
            std::size_t i = 0;
            for (auto& each: initlist) { data[i] = each; i++; }
        }

        // Vectorized arithmetic operations of ColVector4d
        ColVector4d operator+(const ColVector4d& rhs) const;
        ColVector4d operator-(const ColVector4d& rhs) const;
        ColVector4d operator*(double number) const;
        ColVector4d operator/(double number) const;

        ColVector4d& operator+=(const ColVector4d& rhs) { *this = *this + rhs; return *this; }
        ColVector4d& operator-=(const ColVector4d& rhs) { *this = *this - rhs; return *this; }
        ColVector4d& operator*=(double number) { *this = *this * number; return *this; }
        ColVector4d& operator/=(double number) { *this = *this / number; return *this; }
    };

    struct RowVector4d : public BaseMatrix<double,1,4,256>
    {
        RowVector4d() {}
        RowVector4d(double object) : BaseMatrix<double,1,4,256>(object) {}
        RowVector4d(std::initializer_list<double> initlist)
        {
            assert((initlist.size() == 4)); // TODO: Add possible check in compile time via static_assert
            std::size_t i = 0;
            for (auto& each: initlist) { data[i] = each; i++; }
        }

        // Vectorized arithmetic operations of RowVector4d
        RowVector4d operator+(const RowVector4d& rhs) const;
        RowVector4d operator-(const RowVector4d& rhs) const;
        RowVector4d operator*(double number) const;
        RowVector4d operator/(double number) const;

        RowVector4d& operator+=(const RowVector4d& rhs) { *this = *this + rhs; return *this; }
        RowVector4d& operator-=(const RowVector4d& rhs) { *this = *this - rhs; return *this; }
        RowVector4d& operator*=(double number) { *this = *this * number; return *this; }
        RowVector4d& operator/=(double number) { *this = *this / number; return *this; }
    };
}

template <typename Type, std::size_t N, std::size_t M, std::size_t alignment>
__SG_CONSTEXPR_MATRIX__
SG::BaseMatrix<Type,N,M,alignment>::BaseMatrix(const Type& object)
{
    for (auto& each: data) { each = object; }
}

template <typename Type, std::size_t N, std::size_t M, std::size_t alignment>
__SG_CONSTEXPR_MATRIX__
SG::BaseMatrix<Type,N,M,alignment>::BaseMatrix(std::initializer_list<std::initializer_list<Type>> initlist)
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

template <typename Type, std::size_t N, std::size_t M, std::size_t alignment>
std::ostream& operator<<(std::ostream& os, const SG::BaseMatrix<Type,N,M,alignment>& object)
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

template <typename Type, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,N,M> SG::Matrix<Type,N,M>::operator+(const SG::Matrix<Type,N,M>& rhs) const
{
    SG::Matrix<Type,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<Type,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<Type,N,M>::data[i] + rhs.data[i];
    }
    return result;
}

template <typename Type, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,N,M> SG::Matrix<Type,N,M>::operator-(const SG::Matrix<Type,N,M>& rhs) const
{
    SG::Matrix<Type,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<Type,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<Type,N,M>::data[i] - rhs.data[i];
    }
    return result;
}

template <typename Type, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,N,M> SG::Matrix<Type,N,M>::operator*(const Type& number) const
{
    SG::Matrix<Type,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<Type,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<Type,N,M>::data[i] * number;
    }
    return result;
}

template <typename Type, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,N,M> SG::Matrix<Type,N,M>::operator/(const Type& number) const
{
    SG::Matrix<Type,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<Type,N,M>::bufsize; ++i)
    {
        result.data[i] = SG::BaseMatrix<Type,N,M>::data[i] / number;
    }
    return result;
}

// Multiplication on a number on the right hand side
template <typename Type, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,N,M> operator*(const Type& number, const SG::Matrix<Type,N,M>& object)
{
    SG::Matrix<Type,N,M> result {};
    for (std::size_t i=0; i<SG::Matrix<Type,N,M>::bufsize; ++i)
    {
        result.data[i] = number * object.data[i];
    }
    return result;
}

// The matrix multiplication; note that it requires three dimensions
template <typename Type, std::size_t N, std::size_t P, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,N,M> operator*(const SG::Matrix<Type,N,P>& object1, const SG::Matrix<Type,P,M>& object2) 
{
    SG::Matrix<Type,N,M> result {};
    for (std::size_t i=0; i<N; ++i)
    {
        for (std::size_t j=0; j<M; ++j)
        {
            Type tmp = object1(i,0) * object2(0,j);
            for (std::size_t k=1; k<P; ++k) { tmp += object1(i,k) * object2(k,j); }
            result(i,j) = tmp;
        }
    }
    return result;
}

template <typename Type, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,N,1> SG::Matrix<Type,N,M>::col(std::size_t j) const
{
    SG::Matrix<Type,N,1> result {};
    for (std::size_t i=0; i<N; ++i) { result.data[i] = SG::BaseMatrix<Type,N,M>::data[i*4 + j]; }
    return result;
}

template <typename Type, std::size_t N, std::size_t M>
__SG_CONSTEXPR_MATRIX__
SG::Matrix<Type,1,M> SG::Matrix<Type,N,M>::row(std::size_t i) const
{
    SG::Matrix<Type,1,M> result {};
    for (std::size_t j=0; i<M; ++i) { result.data[i] = SG::BaseMatrix<Type,N,M>::data[i*4 + j]; }
    return result;
}

// Vectorized arithmetic operations on Matrix4x4d
// (All are made explicitly inline for optimization)

inline SG::Matrix4x4d
SG::Matrix4x4d::operator+(const SG::Matrix4x4d& rhs) const
{
    SG::Matrix4x4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements, _rhs_elements, _res_elements;
        for (std::size_t i=0; i<4; ++i)
        {
            _lhs_elements = _mm256_load_pd(data + i*4);
            _rhs_elements = _mm256_load_pd(rhs.data + i*4);
            _res_elements = _mm256_add_pd(_lhs_elements, _rhs_elements);
            _mm256_store_pd(result.data + i*4, _res_elements);
        }
    #else
        for (std::size_t i=0; i<16; ++i) { result.data[i] = data[i] + rhs.data[i]; }
    #endif
    return result;
}

inline SG::Matrix4x4d
SG::Matrix4x4d::operator-(const SG::Matrix4x4d& rhs) const
{
    SG::Matrix4x4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements, _rhs_elements, _res_elements;
        for (std::size_t i=0; i<4; ++i)
        {
            _lhs_elements = _mm256_load_pd(data + i*4);
            _rhs_elements = _mm256_load_pd(rhs.data + i*4);
            _res_elements = _mm256_sub_pd(_lhs_elements, _rhs_elements);
            _mm256_store_pd(result.data + i*4, _res_elements);
        }
    #else
        for (std::size_t i=0; i<16; ++i) { result.data[i] = data[i] - rhs.data[i]; }
    #endif
    return result;
}

inline SG::Matrix4x4d
SG::Matrix4x4d::operator*(double number) const
{
    SG::Matrix4x4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements, _rhs_elements, _res_elements;
        _rhs_elements = _mm256_set1_pd(number);
        for (std::size_t i=0; i<4; ++i)
        {
            _lhs_elements = _mm256_load_pd(data + i*4);
            _res_elements = _mm256_mul_pd(_lhs_elements, _rhs_elements);
            _mm256_store_pd(result.data + i*4, _res_elements);
        }
    #else
        for (std::size_t i=0; i<16; ++i) { result.data[i] = data[i] * number; }
    #endif
    return result;
}

inline SG::Matrix4x4d
SG::Matrix4x4d::operator/(double number) const
{
    SG::Matrix4x4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements, _rhs_elements, _res_elements;
        _rhs_elements = _mm256_set1_pd(number);
        for (std::size_t i=0; i<4; ++i)
        {
            _lhs_elements = _mm256_load_pd(data + i*4);
            _res_elements = _mm256_div_pd(_lhs_elements, _rhs_elements);
            _mm256_store_pd(result.data + i*4, _res_elements);
        }
    #else
        for (std::size_t i=0; i<16; ++i) { result.data[i] = data[i] / number; }
    #endif
    return result;
}

// Finally, vectorized version of matrix multiplication
inline SG::Matrix4x4d
operator*(const SG::Matrix4x4d& matrix1, const SG::Matrix4x4d& matrix2)
{
    SG::Matrix4x4d result {};
    #ifdef __AVX2__
        // First, let's make a copy of the matrix with transposed entries
        alignas(256) double buffer[16];
        for (std::size_t i=0; i<4; ++i)
        {
            for (std::size_t j=0; j<4; ++j) { buffer[i*4 + j] = matrix2.data[j*4 + i]; }
        }
        // Then, all the normal part
        __m256d _lhs_elements, _rhs_elements, _res_elements;
        __m128d _values_low, _values_high;
        for (std::size_t i=0; i<4; ++i)
        {
            for (std::size_t j=0; j<4; ++j)
            {
                _lhs_elements = _mm256_load_pd(matrix1.data + i*4);
                _rhs_elements = _mm256_load_pd(buffer + j*4);
                _res_elements = _mm256_mul_pd(_lhs_elements, _rhs_elements);
                // Then, add up the values: reduce to 2 doubles
                _values_low  = _mm256_castpd256_pd128(_res_elements); // Obtain low 2 doubles
                _values_high = _mm256_extractf128_pd(_res_elements, 1); // Obtain high 2 doubles
                _values_low  = _mm_add_pd(_values_low, _values_high); // Add them up and save in a low-128b register
                // And then, let's then sum up two values
                // This saves high 64-bit part in two halfs of _values_high - so then we need to sum up it with the low part
                _values_high = _mm_unpackhi_pd(_values_low, _values_low);
                // Add with the low 64-bit double and obtain it
                result.data[i*4+j] = _mm_cvtsd_f64(_mm_add_sd(_values_low, _values_high));
            }
        }
    #else 
        for (std::size_t i=0; i<4; ++i)
        {
            for (std::size_t j=0; j<4; ++j)
            {
                double tmp = 0;
                for (std::size_t k=0; k<4; ++k)
                { tmp += matrix1.data[i*4 + k]*matrix2.data[k*4 + j]; }
                result.data[i*4 + j] = tmp;
            }
        }
    #endif
    return result;
}

inline SG::Matrix4x4d
SG::commutator(const SG::Matrix4x4d& matrix1, const SG::Matrix4x4d& matrix2)
{
    return matrix1*matrix2 - matrix2*matrix1;
}

SG::Matrix4x4d SG::transpose(const SG::Matrix4x4d& matrix)
{
    SG::Matrix4x4d result {};
    for (std::size_t i=0; i<4; ++i)
    {
        for (std::size_t j=0; j<4; ++j) { result.data[i*4+j] = matrix.data[j*4+i]; }
    }
    return result;
}

SG::Matrix4x4d SG::Matrix4x4d::T() const { return SG::transpose(*this); }

// Vectorized arithmetic operations of ColVector4d
// (All are made explicitly inline for optimization)

inline SG::ColVector4d
SG::ColVector4d::operator+(const SG::ColVector4d& rhs) const
{
    SG::ColVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_load_pd(data);
        __m256d _rhs_elements = _mm256_load_pd(rhs.data);
        __m256d _res_elements = _mm256_add_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] + rhs.data[i]; }
    #endif
    return result;
}

inline SG::ColVector4d
SG::ColVector4d::operator-(const SG::ColVector4d& rhs) const
{
    SG::ColVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_load_pd(data);
        __m256d _rhs_elements = _mm256_load_pd(rhs.data);
        __m256d _res_elements = _mm256_sub_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] - rhs.data[i]; }
    #endif
    return result;
}

inline SG::ColVector4d
SG::ColVector4d::operator*(double number) const
{
    SG::ColVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_load_pd(data);
        __m256d _rhs_elements = _mm256_set1_pd(number);
        __m256d _res_elements = _mm256_mul_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else 
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] * number; }
    #endif
    return result;
}

inline SG::ColVector4d
SG::ColVector4d::operator/(double number) const
{
    SG::ColVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_load_pd(data);
        __m256d _rhs_elements = _mm256_set1_pd(number);
        __m256d _res_elements = _mm256_div_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] / number; }
    #endif
    return result;
}

// Vectorized arithmetic operations of RowVector4d
// (All are made explicitly inline for optimization)

inline SG::RowVector4d
SG::RowVector4d::operator+(const SG::RowVector4d& rhs) const
{
    SG::RowVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_loadu_pd(data);
        __m256d _rhs_elements = _mm256_loadu_pd(rhs.data);
        __m256d _res_elements = _mm256_add_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] + rhs.data[i]; }
    #endif
    return result;
}

inline SG::RowVector4d
SG::RowVector4d::operator-(const SG::RowVector4d& rhs) const
{
    SG::RowVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_loadu_pd(data);
        __m256d _rhs_elements = _mm256_loadu_pd(rhs.data);
        __m256d _res_elements = _mm256_sub_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] - rhs.data[i]; }
    #endif
    return result;
}

inline SG::RowVector4d
SG::RowVector4d::operator*(double number) const
{
    SG::RowVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_loadu_pd(data);
        __m256d _rhs_elements = _mm256_set1_pd(number);
        __m256d _res_elements = _mm256_mul_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else 
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] * number; }
    #endif
    return result;
}

inline SG::RowVector4d
SG::RowVector4d::operator/(double number) const
{
    SG::RowVector4d result {};
    #ifdef __AVX2__
        __m256d _lhs_elements = _mm256_loadu_pd(data);
        __m256d _rhs_elements = _mm256_set1_pd(number);
        __m256d _res_elements = _mm256_div_pd(_lhs_elements, _rhs_elements);
        _mm256_store_pd(result.data, _res_elements);
    #else
        for (std::size_t i=0; i<4; ++i) { result.data[i] = data[i] / number; }
    #endif
    return result;
}

#endif

// TODOs:
// 1. Realise multiplication of a matrix on a vector, and the dot product
// 2. SSE/AVX for N=4, M=4 cases (but for what types?... seems, for doubles.)
// 3. Exponential function (and here we will see possible growth in performance)
// 4. Fundamental bases, identity matrices...