//==================================================================================================
//  File:       linear_algebra.hpp
//
//  Summary:    This is a driver header for including all of the linear algebra facilities
//              defined by the library.
//==================================================================================================

#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

//- STL includes
#include <algorithm>
#include <array>
#if __has_include( <concepts> )
#include <concepts>
#endif
#include <complex>
#include <cstddef>
#if __has_include( <execution> )
#include <execution>
#endif
#include <exception>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#if __has_include( <ranges> )
#include <ranges>
#endif
#if __has_include( <span> )
#include <span>
#endif
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <valarray>

//- mdspan include
#include <experimental/mdspan>
#include "mdspan_extensions/rank_one_extents_specialization.hpp"

//- Implementation headers
#include "linalg/config.hpp"
#include "linalg/macros.hpp"
#include "linalg/private_support.hpp"
#include "linalg/tensor_concepts.hpp"
#include "linalg/forward_declarations.hpp"
#include "linalg/tensor_expression/tensor_expression_traits.hpp"
#include "linalg/tensor_expression/accessor_result.hpp"
#include "linalg/tensor_expression/allocator_result.hpp"
#include "linalg/tensor_expression/layout_result.hpp"
#include "linalg/tensor_expression/unary_tensor_expressions.hpp"
#include "linalg/tensor_expression/binary_tensor_expressions.hpp"
#include "linalg/subtensor.hpp"
#include "linalg/tensor_memory.hpp"
#include "linalg/dr_tensor.hpp"
#include "linalg/fs_tensor.hpp"
#include "linalg/subtensor.hpp"
#include "linalg/arithmetic_operators.hpp"

#endif  //- LINEAR_ALGEBRA_HPP
