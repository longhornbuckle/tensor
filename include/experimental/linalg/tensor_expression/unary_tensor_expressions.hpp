//==================================================================================================
//  File:       unary_tensor_expressions.hpp
//
//  Summary:    This header defines a unary tensor expressions:
//              negate_tensor_expression< Tensor >
//              transpose_tensor_expression< Tensor >
//              conjugate_tensor_expression< Tensor >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{

// Negation tensor expression
template < tensor_expression Tensor >
class negate_tensor_expression
{
  private:
    using self_type = negate_tensor_expression< Tensor >;
  public:
    // Special member functions
    constexpr negate_tensor_expression( Tensor&& t ) noexcept : t_(t) { }
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
        // TBD
      }
      else
      {
        return tensor< value_type,
                       extents_type,
                       layout_result_t< self_type >,
                       typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                       accessor_result_t< self_type > >
          ( *this, allocator_result< self_type >::get_allocator( *this ) );
      }
    };

  private:
    // Data
    Tensor& t_;
};

// Scalar Pre-Multiply
template < class ValueType, tensor_expression Tensor >
class scalar_preprod_tensor_expression
  requires requires ( const ValueType& v1, const typename Tensor::value_type& v2 ) { { v1 * v2; } }
{

};

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP
