//==================================================================================================
//  File:       arithmetic_operators.hpp
//
//  Summary:    This header defines the overloaded operators that implement basic arithmetic
//              operations on vectors, matrices, and tensors.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP
#define LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP

#include <experimental/linear_algebra.hpp>

// Bring operators into std namespace for use with mdspan
namespace  std
{
  using LINALG::operator -;
  using LINALG::operator +;
  using LINALG::operator *;
  using LINALG::operator /;
  using LINALG::operator %;
  using LINALG::operator -=;
  using LINALG::operator +=;
  using LINALG::operator *=;
  using LINALG::operator /=;
  using LINALG::operator %=;
  using LINALG::trans;
  using LINALG::conj;
  using LINALG::inner_prod;
  using LINALG::outer_prod;
} // std namespace

#endif  //- LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP
