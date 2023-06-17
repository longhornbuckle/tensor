//==================================================================================================
//  File:       addition.hpp
//
//  Summary:    This header defines:
//              LINALG::accessor_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
//              LINALG::allocator_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
//              LINALG::layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
//              LINALG::is_alias_assignable< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
//              LINALG_EXPRESSIONS_DETAIL::addition_tensor_expression_traits< FirstTensor, SecondTensor >
//              LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor >
//              LINALG::operator + ( const T1& t1, const T2& t2 )
//              LINALG::operator += ( T1& t1, const T2& t2 )
//==================================================================================================

#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_ADDITION_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_ADDITION_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//-------------------
//  Accessor Result
//-------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstTensor, class SecondTensor >
struct accessor_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< FirstTensor >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< FirstTensor >::value_type >() + ::std::declval< typename ::std::remove_reference_t< SecondTensor >::value_type >() ) >;
};

#else

template < class FirstTensor, class SecondTensor, class Enable >
struct accessor_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor, Enable > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< FirstTensor >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< FirstTensor >::value_type >() + ::std::declval< typename ::std::remove_reference_t< SecondTensor >::value_type >() ) >;
};

#endif

//--------------------
//  Allocator Result
//--------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondTensor > > )
struct allocator_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< FirstTensor > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< SecondTensor > >,
                                                                FirstTensor,
                                                                SecondTensor > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor >& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< FirstTensor > > ||
                   ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< SecondTensor > > )
    {
      return allocator_result< FirstTensor >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< SecondTensor >::get_allocator( t.second() );
    }
  }
};

#else

template < class FirstTensor, class SecondTensor, class Enable >
struct allocator_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor, Enable > >
{
private:
  using T = LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor, Enable >;
public:
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< FirstTensor > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< SecondTensor > >,
                                                                FirstTensor,
                                                                SecondTensor > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const T& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< FirstTensor > > ||
                    ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< SecondTensor > > )
    {
      return allocator_result< FirstTensor >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< SecondTensor >::get_allocator( t.second() );
    }
  }
};

#endif

//-----------------
//  Layout Result
//-----------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondTensor > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_stride > ) &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, typename ::std::remove_reference_t< SecondTensor >::layout_type > &&
                                       ! ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_stride >,
                                     typename ::std::remove_reference_t< FirstTensor >::layout_type,
                                     LINALG::default_layout >;
};

template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondTensor > > )
struct layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::addition_tensor_expression< decltype( ::std::declval< FirstTensor >().operator auto() ), SecondTensor > > >::type;
};

template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondTensor > > )
struct layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, decltype( ::std::declval< SecondTensor >().operator auto() ) > > >::type;
};

template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondTensor > > )
struct layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::addition_tensor_expression< decltype( ::std::declval< FirstTensor >().operator auto() ), decltype( ::std::declval< SecondTensor >().operator auto() ) > > >::type;
};

#else

template < class FirstTensor, class SecondTensor, class Enable >
struct layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor, Enable > >
{
private:
  template < class T, class U >
  struct invalid_helper;
  template < class T, class U >
  struct readable_readable_helper
  {
    using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< T >::layout_type, typename ::std::remove_reference_t< U >::layout_type > &&
                                         ! ::std::is_same_v< typename ::std::remove_reference_t< T >::layout_type, ::std::layout_stride >,
                                       typename ::std::remove_reference_t< T >::layout_type,
                                       LINALG::default_layout >;
  };
  template < class T, class U >
  struct readable_unevaluated_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
  };
  template < class T, class U >
  struct unevaluated_unevaluated_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::addition_tensor_expression< decltype( ::std::declval< FirstTensor >().operator auto() ), decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
  };
public:
  using type = typename ::std::conditional_t< ( ( ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_stride > ) &&
                                                ( ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_stride > ) ),
                                              typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< FirstTensor > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< SecondTensor > >,
                                                                                          readable_readable_helper< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondTensor > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > >,
                                                                                                                invalid_helper< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< FirstTensor > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< SecondTensor > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< SecondTensor >, ::std::remove_reference_t< FirstTensor > >,
                                                                                                                ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondTensor > >,
                                                                                                                                      unevaluated_unevaluated_helper< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > >,
                                                                                                                                      invalid_helper< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > > >,
                                                                                          invalid_helper< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > > >,
                                              invalid_helper< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > >::type;
};

#endif

//----------------------
//  Is Alias Assignable
//-----------------------
template < class FirstTensor, class SecondTensor >
struct is_alias_assignable< LINALG_EXPRESSIONS::addition_tensor_expression< FirstTensor, SecondTensor > > :
  public ::std::conditional_t< 
#ifdef LINALG_ENABLE_CONCEPTS
                               LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstTensor > >,
#else
                               LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< FirstTensor > >,
#endif
                               ::std::conditional_t< 
#ifdef LINALG_ENABLE_CONCEPTS
                                                     LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondTensor > >,
#else
                                                     LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondTensor > >,
#endif
                                                     ::std::integral_constant< bool, is_alias_assignable_v< FirstTensor > && is_alias_assignable_v< SecondTensor > >,
                                                     is_alias_assignable< FirstTensor > >,
                               ::std::conditional_t< 
#ifdef LINALG_ENABLE_CONCEPTS
                                                     LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondTensor > >,
#else
                                                     LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondTensor > >,
#endif
                                                     is_alias_assignable< SecondTensor >,
                                                     ::std::true_type > >
{ };

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

// Addition tensor expression
template < class FirstTensor, class SecondTensor >
class addition_tensor_expression_traits
{
  public:
    // Aliases
    using value_type   = decltype( ::std::declval< typename ::std::remove_reference_t< FirstTensor >::value_type >() + ::std::declval< typename ::std::remove_reference_t< SecondTensor >::value_type >() );
    using index_type   = ::std::common_type_t< typename ::std::remove_reference_t< FirstTensor >::index_type, typename ::std::remove_reference_t< SecondTensor >::index_type >;
    using size_type    = ::std::common_type_t< typename ::std::remove_reference_t< FirstTensor >::size_type, typename ::std::remove_reference_t< SecondTensor >::size_type >;
    using extents_type = ::std::conditional_t< ( ::std::remove_reference_t< FirstTensor >::extents_type::rank_dynamic() == 0 ) || ( ::std::remove_reference_t< SecondTensor >::extents_type::rank_dynamic() != 0 ), typename ::std::remove_reference_t< FirstTensor >::extents_type, typename ::std::remove_reference_t< SecondTensor >::extents_type >;
    using rank_type    = ::std::common_type_t< typename ::std::remove_reference_t< FirstTensor >::rank_type, typename ::std::remove_reference_t< SecondTensor >::rank_type >;
};

LINALG_EXPRESSIONS_DETAIL_END // expression detail namespace

LINALG_EXPRESSIONS_BEGIN // expressions namespace

#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondTensor > > &&
             ( ::std::remove_reference_t< FirstTensor >::rank() == ::std::remove_reference_t< SecondTensor >::rank() ) &&
             LINALG_DETAIL::extents_may_be_equal_v< typename ::std::remove_reference_t< FirstTensor >::extents_type, typename ::std::remove_reference_t< SecondTensor >::extents_type > &&
             requires ( typename ::std::remove_reference_t< FirstTensor >::value_type v1, typename ::std::remove_reference_t< SecondTensor >::value_type v2 ) { v1 + v2; } )
class addition_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< addition_tensor_expression< FirstTensor, SecondTensor >,
                                                                   LINALG_EXPRESSIONS_DETAIL::addition_tensor_expression_traits< FirstTensor, SecondTensor > >
#else
template < class FirstTensor, class SecondTensor, typename Enable >
class addition_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< addition_tensor_expression< FirstTensor, SecondTensor, Enable >,
                                                                   LINALG_EXPRESSIONS_DETAIL::addition_tensor_expression_traits< FirstTensor, SecondTensor > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = addition_tensor_expression< FirstTensor, SecondTensor >;
    #else
    using self_type   = addition_tensor_expression< FirstTensor, SecondTensor, Enable >;
    #endif
    using traits_type = LINALG_EXPRESSIONS_DETAIL::addition_tensor_expression_traits< FirstTensor, SecondTensor >;
    using base_type   = LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr addition_tensor_expression( FirstTensor&& t1, SecondTensor&& t2 )
      noexcept( LINALG_DETAIL::extents_are_equal_v< typename ::std::remove_reference_t< FirstTensor >::extents_type, typename ::std::remove_reference_t< SecondTensor >::extents_type > ) : t1_(t1), t2_(t2)
    {
      if constexpr ( !LINALG_DETAIL::extents_are_equal_v< typename ::std::remove_reference_t< FirstTensor >::extents_type, typename ::std::remove_reference_t< SecondTensor >::extents_type > )
      {
        if ( t1.extents() != t2.extents() ) LINALG_UNLIKELY
        {
          throw length_error( "Tensor extents are incompatable." );
        }
      }
    }
    constexpr addition_tensor_expression& operator = ( const addition_tensor_expression& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
    constexpr addition_tensor_expression& operator = ( addition_tensor_expression&& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
    // Aliases
    using value_type     = typename traits_type::value_type;
    using index_type     = typename traits_type::index_type;
    using size_type      = typename traits_type::size_type;
    using extents_type   = typename traits_type::extents_type;
    using rank_type      = typename traits_type::rank_type;
  private:
    template < class T, bool >
    struct helper
    {
      using type = LINALG::fs_tensor< typename T::value_type,
                                      typename T::extents_type,
                                      LINALG::layout_result_t< T >,
                                      LINALG::accessor_result_t< T > >;
    };
    template < class T >
    struct helper< T, false >
    {
      using type = LINALG::dr_tensor< typename T::value_type,
                                      typename T::extents_type,
                                      LINALG::layout_result_t< T >,
                                      typename T::extents_type,
                                      typename ::std::allocator_traits< LINALG::allocator_result_t< T > >::template rebind_alloc< typename T::value_type >,
                                      LINALG::accessor_result_t< T > >;
    };
  public:
    using evaluated_type = typename helper< self_type, ( extents_type::rank_dynamic() == 0 ) >::type;
    // using evaluated_type = ::std::conditional_t< ( extents_type::rank_dynamic() == 0 ),
    //                                              LINALG::fs_tensor< value_type,
    //                                                                 extents_type,
    //                                                                 LINALG::layout_result_t< self_type >,
    //                                                                 LINALG::accessor_result_t< self_type > >,
    //                                              LINALG::dr_tensor< value_type,
    //                                                                 extents_type,
    //                                                                 LINALG::layout_result_t< self_type >,
    //                                                                 extents_type,
    //                                                                 typename ::std::allocator_traits< LINALG::allocator_result_t< self_type > >::template rebind_alloc< value_type >,
    //                                                                 LINALG::accessor_result_t< self_type > > >;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< FirstTensor >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { if constexpr ( ( ::std::remove_reference_t< FirstTensor >::extents_type::rank_dynamic() == 0 ) || ( ::std::remove_reference_t< SecondTensor >::extents_type::rank_dynamic() != 0 ) ) { return this->t1_.extents(); } else { return this->t2_.extents(); } }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { if constexpr ( ( ::std::remove_reference_t< FirstTensor >::extents_type::rank_dynamic() == 0 ) || ( ::std::remove_reference_t< SecondTensor >::extents_type::rank_dynamic() != 0 ) ) { return this->t1_.extent(n); } else { return this->t2_.extent(n); } }
    // Binary tensor expression function
    [[nodiscard]] constexpr const FirstTensor& first() const noexcept { return this->t1_; }
    [[nodiscard]] constexpr const SecondTensor& second() const noexcept { return this->t2_; }
    // Addition
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( this->t1_, indices ... ) + LINALG_DETAIL::access( this->t2_, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) + LINALG_DETAIL::access( this->t2_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( this->t1_, indices ... ) + LINALG_DETAIL::access( this->t2_, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) + LINALG_DETAIL::access( this->t2_, indices ... ); }
    #endif
  private:
    // Define noexcept specification of conversion operator
    [[nodiscard]] static inline constexpr bool conversion_is_noexcept() noexcept
    {
      if constexpr( extents_type::rank_dynamic() == 0 )
      {
        return ::std::is_nothrow_constructible_v< evaluated_type,
                                                  const base_type& >;
      }
      else
      {
        return ::std::is_nothrow_constructible_v< evaluated_type,
                                                  const base_type&,
                                                  decltype( LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
      }
    }
  public:
    // Implicit conversion
    [[nodiscard]] constexpr operator evaluated_type() const noexcept( conversion_is_noexcept() )
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return evaluated_type( *static_cast< const base_type* >( this ) );
      }
      else
      {
        return evaluated_type( *static_cast< const base_type* >( this ), LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( *this ) ) );
      }
    }
  private:
    // Data
    FirstTensor&  t1_;
    SecondTensor& t2_;
};

LINALG_EXPRESSIONS_END // end expressions namespace

LINALG_BEGIN // linalg namespace

//-----------------------------
//  Binary addition operators
//-----------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class T1, class T2 >
#else
template < class T1, class T2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< const T1&, const T2& >, const T1&, const T2& > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator + ( const T1& t1, const T2& t2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< const T1&, const T2& >, const T1&, const T2& >
#endif
{
  return LINALG_EXPRESSIONS::addition_tensor_expression< const T1&, const T2& >( t1, t2 );
}

//----------------------------------------
//  Binary addition assignment operators
//----------------------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class T1, class T2 >
#else
template < class T1, class T2,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& >, T1&, const T2& > &&
                                            ::std::is_assignable_v< T1&, LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& > > ) > >
#endif
inline constexpr T1&
operator += ( T1& t1, const T2& t2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& >, T1&, const T2& > &&
             ::std::is_assignable_v< T1&, LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& > > )
#endif
{
  return t1 = LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& >( t1, t2 );
}

LINALG_END // linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_ADDITION_HPP
