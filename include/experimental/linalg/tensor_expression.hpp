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
class negate_tensor_expression
{
  public:
    // Special member functions
    constexpr negate_tensor_expression( const Tensor t ) noexcept : t_t(t) { }
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
    [[nodiscard]] constexpr Tensor& underlying() noexcept { return this->t_; }
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


  private:
    // Data
    Tensor t_;
};

template < tensor_expression TensorA, tensor_expression TensorB >
class binary_tensor_expression
{
};

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_HPP
