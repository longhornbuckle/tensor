//==================================================================================================
//  File:       unary_base.hpp
//
//  Summary:    This header defines:
//              LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< Tensor, Traits >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_UNARY_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_UNARY_HPP

#include <experimental/linear_algebra.hpp>

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

template < class Tensor, class Traits >
class unary_tensor_expression_base;

LINALG_EXPRESSIONS_DETAIL_END // expressions detail namespace

LINALG_BEGIN // linalg namespace

template < class Tensor, class Traits >
struct is_alias_assignable< LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< Tensor, Traits > > :
  public is_alias_assignable< Tensor > { };

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class > class UTE,
                              class Tensor,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor >, Traits >
#else
template < template < class, class > class UTE,
                                     class Tensor,
                                     class Enable,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor, Enable >, Traits >
#endif
{
  private :
    #ifdef LINALG_ENABLE_CONCEPTS
    using expression_type = UTE< Tensor >;
    #else
    using expression_type = UTE< Tensor, Enable >;
    #endif
    using traits_type     = Traits;
  public:
    // Aliases
    using value_type   = typename traits_type::value_type;
    using index_type   = typename traits_type::index_type;
    using size_type    = typename traits_type::size_type;
    using extents_type = typename traits_type::extents_type;
    using rank_type    = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return expression_type::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return static_cast< const expression_type* >(this)->extents(); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { return static_cast< const expression_type* >(this)->extent( n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr decltype(auto) underlying() const noexcept { return static_cast< const expression_type* >(this)->underlying(); }
    // Index access
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator[]( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator[]( indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator()( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator()( indices ... ); }
    #endif
    // Evaluated expression
    [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION auto evaluate() const noexcept( this->UTE< Tensor >::conversion_is_noexcept() )
    {
      return this->UTE< Tensor >::evaluate();
    }
};

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class, class > class UTE,
                                     class Tensor,
                                     class Param,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor, Param >, Traits >
#else
template < template < class, class, class > class UTE,
                                            class Tensor,
                                            class Param,
                                            class Enable,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor, Param, Enable >, Traits >
#endif
{
  private :
    #ifdef LINALG_ENABLE_CONCEPTS
    using expression_type = UTE< Tensor, Param >;
    #else
    using expression_type = UTE< Tensor, Param, Enable >;
    #endif
    using traits_type     = Traits;
  public:
    // Aliases
    using value_type   = typename traits_type::value_type;
    using index_type   = typename traits_type::index_type;
    using size_type    = typename traits_type::size_type;
    using extents_type = typename traits_type::extents_type;
    using rank_type    = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return expression_type::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return static_cast< const expression_type* >(this)->extents(); }
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept { return static_cast< const expression_type* >(this)->extent( n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr decltype(auto) underlying() const noexcept { return static_cast< const expression_type* >(this)->underlying(); }
    // Index access
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator[]( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator[]( indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator()( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator()( indices ... ); }
    #endif
    // Evaluated expression
    [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION auto evaluate() const noexcept( this->expression_type::conversion_is_noexcept() )
    {
      return this->expression_type::evaluate();
    }
};

LINALG_EXPRESSIONS_DETAIL_END // expressions namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_UNARY_BASE_HPP
