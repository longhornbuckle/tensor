//==================================================================================================
//  File:       tensor_expression.hpp
//
//  Summary:    This header defines a tensor expressions.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{


template < tensor_expression Tensor >
class tensor_negate_expression;

template < tensor_expression Tensor >
class tensor_transpose_expression;

template < tensor_expression Tensor >
class tensor_conjugate_expression;


template < class Tensor >
struct layout_result;

template < readable_tensor Tensor >
struct layout_result< tensor_negate_expression< Tensor > >
{
  using type = typename Tensor::layout_type;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< tensor_negate_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < readable_tensor Tensor >
struct layout_result< tensor_transpose_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_stride > )
{
  using type = default_layout;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< tensor_transpose_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < readable_tensor Tensor >
struct layout_result< tensor_conjugate_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_stride > )
{
  using type = default_layout;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< tensor_conjugate_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < tensor_expression Tensor >
using layout_result_t = typename layout_result< Tensor >::type;


template < class Tensor >
struct accessor_result;

template < readable_tensor Tensor >
struct accessor_result< tensor_negate_expression< Tensor > >
{
  using type = typename Tensor::accessor_type;
};

template < unevaluated_tensor_expression Tensor >
struct accessor_result< tensor_negate_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < readable_tensor Tensor >
struct accessor_result< tensor_transpose_expression< Tensor > >
{
  using type = typename Tensor::accessor_type;
};

template < unevaluated_tensor_expression Tensor >
struct accessor_result< tensor_conjugate_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < class Tensor >
using accessor_result_t = typename accessor_result< Tensor >::type;


template < class Tensor >
struct allocator_result
{
  using type = ::std::allocator< typename Tensor::value_type >;
};

template < dynamic_tensor Tensor >
struct allocator_result< Tensor >
{
  using type = typename Tensor::allocator_type;
};

template < dynamic_tensor Tensor >
struct allocator_result< tensor_negate_expression< Tensor > >
{
  using type = typename Tensor::allocator_type;
};

template < dynamic_tensor Tensor >
struct allocator_result< tensor_transpose_expression< Tensor > >
{
  using type = typename Tensor::allocator_type;
};

template < dynamic_tensor Tensor >
struct allocator_result< tensor_conjugate_expression< Tensor > >
{
  using type = typename Tensor::allocator_type;
};

template < class Tensor >
using allocator_result_t = typename allocator_result< Tensor >::type;

// Negate tensor expression
template < tensor_expression Tensor >
class negate_tensor_expression
{
  public:
    // Special member functions
    constexpr negate_tensor_expression( Tensor& t ) noexcept : t_t(t) { }
    constexpr negate_tensor_expression& operator = ( const negate_tensor_expression& t ) noexcept { this->t_ = t.t_; }
    constexpr negate_tensor_expression& operator = ( negate_tensor_expression&& t ) noexcept { this->t_ = t.t_; }
    // Aliases
    using value_type   = decltype( - ::std::declval<typename Tensor::value_type>() );
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename Tensor::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return t_.extents(); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return t_.extent(n); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Negate
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( *this, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( *this, indices ... ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto()
    {
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< Tensor >,
                          accessor_result_t< Tensor > >
          ( *this );
      }
      else
      {
        return dr_tensor< value_type,
                          extents_type,
                          layout_result_t< Tensor >,
                          extents_type,
                          typename ::std::allocator_traits< allocator_result_t< Tensor > >::template rebind_t<value_type>,
                          accessor_result_t< Tensor > >
          ( *this );
      }
    };

  private:
    // Data
    Tensor& t_;
};

template < tensor_expression TensorA, tensor_expression TensorB >
class binary_tensor_expression
{
};

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_HPP
