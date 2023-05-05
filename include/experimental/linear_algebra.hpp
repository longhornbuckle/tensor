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
#include <concepts>
#include <complex>
#include <cstddef>
#include <execution>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#ifdef LINALG_ENABLE_RANGED
#include <ranges>
#endif
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <valarray>

//- mdspan include
#include <experimental/mdspan>
#include <experimental/mdspan_extensions/rank_one_extents_specialization.hpp>
using ::std::experimental::dynamic_extent;

// For optimized fixed size tensors
#include <util/fixed_size_allocator.hpp>

//- Implementation headers
#include "linear_algebra/config.hpp"
#include "linear_algebra/macros.hpp"
#include "linear_algebra/private_support.hpp"
#include "linear_algebra/forward_declarations.hpp"
// #include "linear_algebra/tensor_concepts.hpp"
// #include "linear_algebra/vector_concepts.hpp"
// #include "linear_algebra/matrix_concepts.hpp"
#include "linear_algebra/subtensor.hpp"
#include "linear_algebra/tensor_memory.hpp"
#include "linear_algebra/tensor.hpp"
// #include "linear_algebra/instant_evaluated_operations.hpp"
// namespace std::experimental::math::operations { using namespace std::experimental::math::instant_evaluated_operations; }
// #include "linear_algebra/arithmetic_operators.hpp"

#endif  //- LINEAR_ALGEBRA_HPP
