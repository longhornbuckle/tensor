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
class tensor_expression_helper
{
  private:
    template < class T >
    struct layout requires ( !unevaluated_tensor_expression<T> )
    {
      using type = typename T::layout_type;
    };
    template < unevaluated_tensor_expression T >
    struct layout
    {
      using type = typename decltype( ::std::declval<T>().operator auto() )::layout_type;
    };
    template < class T >
    struct accessor requires ( !unevaluated_tensor_expression<T> )
    {
      using type = typename T::accessor_type;
    };
    template < unevaluated_tensor_expression T >
    struct accessor
    {
      using type = typename decltype( ::std::declval<T>().operator auto() )::accessor_type;
    };
    template < class T >
    struct allocator
    {
      using type = ::std::allocator< typename T::value_type >;
    };
    template < class T > requires ( dynamic_tensor< ::std::remove_cv_t< T > > )
    {
      using type = typename ::std::remove_cv_t< T >::allocator_type;
    };
    template < class T > requires ( unevaluated_tensor_expression<T> && dynamic_tensor< decltype( ::std::declval<T>().operator auto() ) > )
    {
      using type = typename decltype( ::std::declval<T>().operator auto() )::allocator_type;
    };
  public:
    using layout_type = typename layout<Tensor>::type;
    using accessor_type = typename accessor<Tensor>::type;
    using allocator_type = typename allocator<Tensor>::type;
};

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
                          typename tensor_expression_helper<Tensor>::layout_type,
                          typename tensor_expression_helper<Tensor>::accessor_type >
          ( *this );
      }
      else
      {
        return dr_tensor< value_type,
                          extents_type,
                          typename tensor_expression_helper<Tensor>::layout_type,
                          typename ::std::allocator_traits< tensor_expression_helper<Tensor>::allocator_type >::template rebind_t<value_type>,
                          typename tensor_expression_helper<Tensor>::accessor_type >
          ( *this );
      }
    };

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
