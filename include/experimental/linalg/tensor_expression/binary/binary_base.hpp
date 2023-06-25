//==================================================================================================
//  File:       binary_base.hpp
//
//  Summary:    This header defines:
//              LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< Tensor, Traits >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_BINARY_BASE_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_BINARY_BASE_HPP

#include <experimental/linear_algebra.hpp>

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

template < class Tensor, class Traits >
class binary_tensor_expression_base;

LINALG_EXPRESSIONS_DETAIL_END // expressions detail namespace

LINALG_BEGIN // linalg namespace

template < class Tensor, class Traits >
struct is_alias_assignable< LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< Tensor, Traits > > :
  public is_alias_assignable< Tensor > { };

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace


#ifdef LINALG_ENABLE_CONCEPTS
template < template < class, class > class BTE,
                                     class FirstTensor,
                                     class SecondTensor,
           class Traits >
class binary_tensor_expression_base< BTE< FirstTensor, SecondTensor >, Traits >
#else
template < template < class, class, class > class BTE,
                                            class FirstTensor,
                                            class SecondTensor,
                                            typename Enable,
           class Traits >
class binary_tensor_expression_base< BTE< FirstTensor, SecondTensor, Enable >, Traits >
#endif
{
  private :
    #ifdef LINALG_ENABLE_CONCEPTS
    using expression_type = BTE< FirstTensor, SecondTensor >;
    #else
    using expression_type = BTE< FirstTensor, SecondTensor, Enable >;
    #endif
    using traits_type     = Traits;
  public:
    // Aliases
    using value_type     = typename traits_type::value_type;
    using index_type     = typename traits_type::index_type;
    using size_type      = typename traits_type::size_type;
    using extents_type   = typename traits_type::extents_type;
    using rank_type      = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return expression_type::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return static_cast< const expression_type* >(this)->extents(); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { return static_cast< const expression_type* >(this)->extent( n ); }
    // Binary tensor expression function
    [[nodiscard]] constexpr decltype(auto) first() const noexcept { return static_cast< const expression_type* >(this)->first(); }
    [[nodiscard]] constexpr decltype(auto) second() const noexcept { return static_cast< const expression_type* >(this)->second(); }
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

LINALG_EXPRESSIONS_DETAIL_END // expressions detail namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_BINARY_BASE_HPP
