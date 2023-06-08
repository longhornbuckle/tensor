//==================================================================================================
//  File:       binary_tensor_expressions.hpp
//
//  Summary:    This header defines a binary tensor expressions:
//              addition_tensor_expression< Tensor >
//              subtraction_tensor_expression< Tensor >
//              scalar_preprod_tensor_expression< S, Tensor >
//              scalar_postprod_tensor_expression< S, Tensor >
//              scalar_division_tensor_expression< S, Tensor >
//              scalar_modulo_tensor_expression< S, Tensor >
//              matrix_product_expression< FirstMatrix, SecondMatrix >
//              vector_matrix_product_expression< Vector, Matrix >
//              matrix_vector_product_expression< Matrix, Vector >
//              outer_product_expression< FirstVector, SecondVector >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_TENSOR_EXPRESSIONS_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_TENSOR_EXPRESSIONS_HPP

#include <experimental/linear_algebra.hpp>

LINALG_EXPRESSIONS_BEGIN // expressions namespace

// // Subtraction tensor expression
// #ifdef LINALG_ENABLE_CONCEPTS
// template < LINALG_CONCEPTS::tensor_expression FirstTensor, LINALG_CONCEPTS::tensor_expression SecondTensor >
//   requires ( ( FirstTensor::rank() == SecondTensor::rank() ) &&
//              LINALG_DETAIL::extents_may_be_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > &&
//              requires ( typename FirstTensor::value_type v1, typename SecondTensor::value_type v2 ) { v1 - v2; } )
// #else
// template < class FirstTensor, class SecondTensor, typename Enable >
// #endif
// class subtraction_tensor_expression
// {
//   private:
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = subtraction_tensor_expression< FirstTensor, SecondTensor >;
//     #else
//     using self_type = subtraction_tensor_expression< FirstTensor, SecondTensor, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr subtraction_tensor_expression( FirstTensor&& t1, SecondTensor&& t2 )
//       noexcept( LINALG_DETAIL::extents_are_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > ) : t1_(t1), t2_(t2)
//     {
//       if constexpr ( !LINALG_DETAIL::extents_are_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > )
//       {
//         if ( t1.extents() != t2.extents() ) LINALG_UNLIKELY
//         {
//           throw length_error( "Tensor extents are incompatable." );
//         }
//       }
//     }
//     constexpr subtraction_tensor_expression& operator = ( const subtraction_tensor_expression& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
//     constexpr subtraction_tensor_expression& operator = ( subtraction_tensor_expression&& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval< typename FirstTensor::value_type >() - ::std::declval< typename SecondTensor::value_type >() );
//     using index_type   = ::std::common_type_t< typename FirstTensor::index_type, typename SecondTensor::index_type >;
//     using size_type    = ::std::common_type_t< typename FirstTensor::size_type, typename SecondTensor::size_type >;
//     using extents_type = ::std::conditional_t< ( FirstTensor::extents_type::rank_dynamic() == 0 ) || ( SecondTensor::extents_type::rank_dynamic() != 0 ), typename FirstTensor::extents_type, typename SecondTensor::extents_type >;
//     using rank_type    = ::std::common_type_t< typename FirstTensor::rank_type, typename SecondTensor::rank_type >;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return FirstTensor::rank(); }
//     [[nodiscard]] constexpr extents_type extents() noexcept { if constexpr ( ( FirstTensor::extents_type::rank_dynamic() == 0 ) || ( SecondTensor::extents_type::rank_dynamic() != 0 ) ) { return this->t1_.extents(); } else { return this->t2_.extents(); } }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { if constexpr ( ( FirstTensor::extents_type::rank_dynamic() == 0 ) || ( SecondTensor::extents_type::rank_dynamic() != 0 ) ) { return this->t1_.extent(n); } else { return this->t2_.extent(n); } }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const FirstTensor& first() const noexcept { return this->t1_; }
//     [[nodiscard]] constexpr const SecondTensor& second() const noexcept { return this->t2_; }
//     // Subtraction
//     #if LINALG_USE_BRACKET_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) - LINALG_DETAIL::access( this->t2_, indices ... ); }
//     #endif
//     #if LINALG_USE_PAREN_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) - LINALG_DETAIL::access( this->t2_, indices ... ); }
//     #endif
//   private:
//     // Define noexcept specification of conversion operator
//     [[nodiscard]] static inline constexpr bool conversion_is_noexcept() noexcept
//     {
//       if constexpr( extents_type::rank_dynamic() == 0 )
//       {
//         return ::std::is_nothrow_constructible_v< LINALG::fs_tensor< value_type,
//                                                                      extents_type,
//                                                                      LINALG::layout_result_t< self_type >,
//                                                                      LINALG::accessor_result_t< self_type > >,
//                                                   const base_type& >;
//       }
//       else
//       {
//         return ::std::is_nothrow_constructible_v< LINALG::dr_tensor< value_type,
//                                                                      extents_type,
//                                                                      LINALG::layout_result_t< self_type >,
//                                                                      extents_type,
//                                                                      typename ::std::allocator_traits< LINALG::allocator_result_t< self_type > >::template rebind_alloc< value_type >,
//                                                                      LINALG::accessor_result_t< self_type > >,
//                                                   const base_type&,
//                                                   decltype( LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
//       }
//     }
//   public:
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() const noexcept( conversion_is_noexcept() )
//     {
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return LINALG::fs_tensor< value_type,
//                                   extents_type,
//                                   LINALG::layout_result_t< self_type >,
//                                   LINALG::accessor_result_t< self_type > >
//           ( *static_cast< const base_type* >( this ) );
//       }
//       else
//       {
//         return LINALG::dr_tensor< value_type,
//                                   extents_type,
//                                   LINALG::layout_result_t< self_type >,
//                                   extents_type,
//                                   typename ::std::allocator_traits< LINALG::allocator_result_t< self_type > >::template rebind_alloc< value_type >,
//                                   LINALG::accessor_result_t< self_type > >
//           ( *static_cast< const base_type* >( this ), LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( *this ) ) );
//       }
//   private:
//     // Data
//     FirstTensor&  t1_;
//     SecondTensor& t2_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < class S, LINALG_CONCEPTS::tensor_expression Tensor >
//   requires requires ( const S& s, const typename Tensor::value_type& v ) { s * v; }
// #else
// template < class S, class Tensor, typename Enable >
// #endif
// class scalar_preprod_tensor_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = scalar_preprod_tensor_expression< S, Tensor >;
//     #else
//     using self_type = scalar_preprod_tensor_expression< S, Tensor, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr scalar_preprod_tensor_expression( S&& s, Tensor&& t ) noexcept : s_(s), t_(t) { };
//     constexpr scalar_preprod_tensor_expression& operator = ( const scalar_preprod_tensor_expression& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     constexpr scalar_preprod_tensor_expression& operator = ( scalar_preprod_tensor_expression&& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<S>() * ::std::declval<typename Tensor::value_type>() );
//     using index_type   = typename Tensor::index_type;
//     using size_type    = typename Tensor::size_type;
//     using extents_type = typename Tensor::extents_type;
//     using rank_type    = typename Tensor::rank_type;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extent(n); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const S& first() const noexcept { return this->s_; }
//     [[nodiscard]] constexpr const Tensor& second() const noexcept { return this->t_; }
//     // Pre-Scalar Multiply
//     #if LINALG_USE_BRACKET_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return this->s_ * LINALG_DETAIL::access( this->t1_, indices ... ); }
//     #endif
//     #if LINALG_USE_PAREN_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return this->s_ * LINALG_DETAIL::access( this->t1_, indices ... ); }
//     #endif
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Data
//     S&      s_;
//     Tensor& t_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < class S, LINALG_CONCEPTS::tensor_expression Tensor >
//   requires requires ( const S& s, const typename Tensor::value_type& v ) { v * s; }
// #else
// template < class S, class Tensor, typename Enable >
// #endif
// class scalar_postprod_tensor_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = scalar_postprod_tensor_expression< Tensor, S >;
//     #else
//     using self_type = scalar_postprod_tensor_expression< Tensor, S, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr scalar_postprod_tensor_expression( Tensor&& t, S&& s ) noexcept : t_(t), s_(s) { };
//     constexpr scalar_postprod_tensor_expression& operator = ( const scalar_postprod_tensor_expression& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     constexpr scalar_postprod_tensor_expression& operator = ( scalar_postprod_tensor_expression&& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<typename Tensor::value_type>() * ::std::declval<S>() );
//     using index_type   = typename Tensor::index_type;
//     using size_type    = typename Tensor::size_type;
//     using extents_type = typename Tensor::extents_type;
//     using rank_type    = typename Tensor::rank_type;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extent(n); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const S& first() const noexcept { return this->s_; }
//     [[nodiscard]] constexpr const Tensor& second() const noexcept { return this->t_; }
//     // Post-Scalar Multiply
//     #if LINALG_USE_BRACKET_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) * this->s_; }
//     #endif
//     #if LINALG_USE_PAREN_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) * this->s_; }
//     #endif
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Data
//     Tensor& t_;
//     S&      s_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < LINALG_CONCEPTS::tensor_expression Tensor, class S >
//   requires requires ( const S& s, const typename Tensor::value_type& v ) { v / s; }
// #else
// template < class Tensor, class S, typename Enable >
// #endif
// class scalar_division_tensor_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = scalar_division_tensor_expression< Tensor, S >;
//     #else
//     using self_type = scalar_division_tensor_expression< Tensor, S, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr scalar_division_tensor_expression( Tensor&& t, S&& s ) noexcept : t_(t), s_(s) { };
//     constexpr scalar_division_tensor_expression& operator = ( const scalar_division_tensor_expression& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     constexpr scalar_division_tensor_expression& operator = ( scalar_division_tensor_expression&& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<typename Tensor::value_type>() / ::std::declval<S>() );
//     using index_type   = typename Tensor::index_type;
//     using size_type    = typename Tensor::size_type;
//     using extents_type = typename Tensor::extents_type;
//     using rank_type    = typename Tensor::rank_type;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extent(n); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const S& first() const noexcept { return this->s_; }
//     [[nodiscard]] constexpr const Tensor& second() const noexcept { return this->t_; }
//     // Post-Scalar Divide
//     #if LINALG_USE_BRACKET_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) / this->s_; }
//     #endif
//     #if LINALG_USE_PAREN_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) / this->s_; }
//     #endif
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                    accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Data
//     Tensor& t_;
//     S&      s_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < LINALG_CONCEPTS::tensor_expression Tensor, class S >
//   requires requires ( const S& s, const typename Tensor::value_type& v ) { v % s; }
// #else
// template < class Tensor, class S, typename Enable >
// #endif
// class scalar_modulo_tensor_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = scalar_modulo_tensor_expression< Tensor, S >;
//     #else
//     using self_type = scalar_modulo_tensor_expression< Tensor, S, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr scalar_modulo_tensor_expression( Tensor&& t, S&& s ) noexcept : t_(t), s_(s) { };
//     constexpr scalar_modulo_tensor_expression& operator = ( const scalar_modulo_tensor_expression& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     constexpr scalar_modulo_tensor_expression& operator = ( scalar_modulo_tensor_expression&& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<typename Tensor::value_type>() % ::std::declval<S>() );
//     using index_type   = typename Tensor::index_type;
//     using size_type    = typename Tensor::size_type;
//     using extents_type = typename Tensor::extents_type;
//     using rank_type    = typename Tensor::rank_type;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extent(n); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const S& first() const noexcept { return this->s_; }
//     [[nodiscard]] constexpr const Tensor& second() const noexcept { return this->t_; }
//     // Post-Scalar Modulo
//     #if LINALG_USE_BRACKET_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) % this->s_; }
//     #endif
//     #if LINALG_USE_PAREN_OPERATOR
//     template < class ... OtherIndexType >
//     [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
//     #ifdef LINALG_ENABLE_CONCEPTS
//       requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
//     #endif
//       { return LINALG_DETAIL::access( this->t1_, indices ... ) % this->s_; }
//     #endif
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                    accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Data
//     Tensor& t_;
//     S&      s_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < LINALG_CONCEPTS::matrix_expression FirstMatrix, LINALG_CONCEPTS::matrix_expression SecondMatrix >
//   requires ( ( requires ( const typename FirstMatrix::value_type& v1, const typename SecondMatrix::value_type& v2 ) { v1 * v2; } ) &&
//              ( ( FirstMatrix::extents_type::static_extent(1) == SecondMatrix::extents_type::static_extent(0) ) ||
//                ( FirstMatrix::extents_type::static_extent(1) == ::std::dynamic_extent ) ||
//                ( SecondMatrix::extents_type::static_extent(0) == ::std::dynamic_extent ) ) )
// #else
// template < class FirstMatrix, class SecondMatrix, typename Enable >
// #endif
// class matrix_product_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = matrix_product_expression< FirstMatrix, SecondMatrix >;
//     #else
//     using self_type = matrix_product_expression< FirstMatrix, SecondMatrix, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr matrix_product_expression( FirstMatrix&& m1, SecondMatrix&& m2 ) noexcept( FirstMatrix::extents_type::static_extent(1) == SecondMatrix::extents_type::static_extent(0) ) :
//       m1_(m1), m2_(m2)
//     {
//       if constexpr ( FirstMatrix::extents_type::static_extent(1) != SecondMatrix::extents_type::static_extent(0) )
//       {
//         if ( FirstMatrix::extents_type::extent(1) != SecondMatrix::extents_type::extent(0) ) LINALG_UNLIKELY
//         {
//           throw length_error( "Matrix extents are incompatible for matrix multiplication." );
//         }
//       }
//     };
//     constexpr matrix_product_expression& operator = ( const matrix_product_expression& t ) noexcept { this->m1_ = t.m1_; this->m2_ = t.m2_; }
//     constexpr matrix_product_expression& operator = ( matrix_product_expression&& t ) noexcept { this->t_ = t.m1_; this->m2_ = t.m2_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<typename FirstMatrix::value_type>() * ::std::declval<typename SecondMatrix::value_type>() );
//     using index_type   = ::std::common_type_t< typename FirstMatrix::index_type, typename SecondMatrix::index_type >;
//     using size_type    = ::std::common_type_t< typename FirstMatrix::size_type, typename SecondMatrix::size_type >;
//     using extents_type = ::std::extents< ::std::common_type_t< typename FirstMatrix::extents_type::size_type, typename SecondMatrix::extents_type::size_type >,
//                                          FirstMatrix::extents_type::static_extent(0),
//                                          SecondMatrix::extents_type::static_extent(1) >;
//     using rank_type    = ::std::common_type_t< typename FirstMatrix::rank_type, typename SecondMatrix::rank_type >;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return 2; }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return extents_type( this->m1_.extent(0), this->m2_.extent(1) ); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return ( n == 0 ) ? this->m1_.extent(0) : this->m2_.extent(n); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const FirstMatrix& first() const noexcept { return this->m1_; }
//     [[nodiscard]] constexpr const SecondMatrix& second() const noexcept { return this->m2_; }
//     // Matrix Product
//     #if LINALG_USE_BRACKET_OPERATOR
//     [[nodiscard]] constexpr value_type operator[]( index_type index1, index_type index2 ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<FirstMatrix>(), index1, ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<SecondMatrix>(), ::std::declval<index_type>(), index2 ) ) )
//     {
//       return this->access( index1, index2 );
//     }
//     #endif
//     [[nodiscard]] constexpr value_type operator()( index_type index1, index_type index2 ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<FirstMatrix>(), index1, ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<SecondMatrix>(), ::std::declval<index_type>(), index2 ) ) )
//     {
//       return this->access( index1, index2 );
//     }
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                    accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Access
//     [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION value_type access( index_type index1, index_type index2 ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<FirstMatrix>(), index1, ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<SecondMatrix>(), ::std::declval<index_type>(), index2 ) ) )
//     {
//       value_type val { 0 };
//       if constexpr ( FirstMatrix::static_extent(1) != ::std::dynamic_extent )
//       {
//         for ( auto count = 0; count < this->m1_.extent(1); ++count )
//         {
//           val += LINALG_DETAIL::access( this->m1_, index1, count ) * LINALG_DETAIL::access( this->m2_, count, index2 );
//         }
//       }
//       else
//       {
//         for ( auto count = 0; count < this->m2_.extent(0); ++count )
//         {
//           val += LINALG_DETAIL::access( this->m1_, index1, count ) * LINALG_DETAIL::access( this->m2_, count, index2 );
//         }
//       }
//       return val;
//     }
//     // Data
//     FirstMatrix& m1_;
//     SecondMatrix& m2_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < LINALG_CONCEPTS::vector_expression Vector, LINALG_CONCEPTS::matrix_expression Matrix >
//   requires ( ( requires ( const typename Vector::value_type& v1, const typename Matrix::value_type& v2 ) { v1 * v2; } ) &&
//              ( ( Vector::extents_type::static_extent(0) == Matrix::extents_type::static_extent(0) ) ||
//                ( Vector::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
//                ( Matrix::extents_type::static_extent(0) == ::std::dynamic_extent ) ) )
// #else
// template < class Vector, class Matrix, typename Enable >
// #endif
// class vector_matrix_product_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = vector_matrix_product_expression< Vector, Matrix >;
//     #else
//     using self_type = vector_matrix_product_expression< Vector, Matrix, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr vector_matrix_product_expression( Vector&& v, Matrix&& m ) noexcept( Vector::extents_type::static_extent(0) == Matrix::extents_type::static_extent(0) ) :
//       v_(v), m_(m)
//     {
//       if constexpr ( Vector::extents_type::static_extent(0) != Matrix::extents_type::static_extent(0) )
//       {
//         if ( Vector::extents_type::extent(0) != Matrix::extents_type::extent(0) ) LINALG_UNLIKELY
//         {
//           throw length_error( "Vector and Matrix extents are incompatible for vector-matrix multiplication." );
//         }
//       }
//     };
//     constexpr vector_matrix_product_expression& operator = ( const vector_matrix_product_expression& t ) noexcept { this->v_ = t.v_; this->m_ = t.m_; }
//     constexpr vector_matrix_product_expression& operator = ( vector_matrix_product_expression&& t ) noexcept { this->v_ = t.v_; this->m_ = t.m_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<typename Vector::value_type>() * ::std::declval<typename Matrix::value_type>() );
//     using index_type   = ::std::common_type_t< typename Vector::index_type, typename Matrix::index_type >;
//     using size_type    = ::std::common_type_t< typename Vector::size_type, typename Matrix::size_type >;
//     using extents_type = ::std::extents< ::std::common_type_t< typename Vector::extents_type::size_type, typename Matrix::extents_type::size_type >,
//                                          Matrix::extents_type::static_extent(1) >;
//     using rank_type    = ::std::common_type_t< typename Vector::rank_type, typename Matrix::rank_type >;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return 1; }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return extents_type( this->m_.extent(1) ); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->m_.extent(n); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const Vector& first() const noexcept { return this->v_; }
//     [[nodiscard]] constexpr const Matrix& second() const noexcept { return this->m_; }
//     // Matrix Product
//     #if LINALG_USE_BRACKET_OPERATOR
//     [[nodiscard]] constexpr value_type operator[]( index_type index ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<Vector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<Matrix>(), ::std::declval<index_type>(), index ) ) )
//     {
//       return this->access( index );
//     }
//     #endif
//     [[nodiscard]] constexpr value_type operator()( index_type index ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<Vector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<Matrix>(), ::std::declval<index_type>(), index ) ) )
//     {
//       return this->access( index );
//     }
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Access
//     [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION value_type access( const index_type& index ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<Vector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<Matrix>(), ::std::declval<index_type>(), index ) ) )
//     {
//       value_type val { 0 };
//       if constexpr ( Vector::extents_type::static_extent(0) != ::std::dynamic_extent )
//       {
//         for ( auto count = 0; count < this->v_.extent(0); ++count )
//         {
//           val += LINALG_DETAIL::access( this->v_, count ) * LINALG_DETAIL::access( this->m_, count, index );
//         }
//       }
//       else
//       {
//         for ( auto count = 0; count < this->m_.extent(0); ++count )
//         {
//           val += LINALG_DETAIL::access( this->v_, count ) * LINALG_DETAIL::access( this->m_, count, index );
//         }
//       }
//       return val;
//     }
//     // Data
//     Vector& v_;
//     Matrix& m_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < LINALG_CONCEPTS::matrix_expression Matrix, LINALG_CONCEPTS::vector_expression Vector >
//   requires ( ( requires ( const typename Matrix::value_type& v1, const typename Vector::value_type& v2 ) { v1 * v2; } ) &&
//              ( ( Vector::extents_type::static_extent(0) == Matrix::extents_type::static_extent(1) ) ||
//                ( Vector::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
//                ( Matrix::extents_type::static_extent(1) == ::std::dynamic_extent ) ) )
// #else
// template < class Matrix, class Vector, typename Enable >
// #endif
// class matrix_vector_product_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = matrix_vector_product_expression< Matrix, Vector >;
//     #else
//     using self_type = matrix_vector_product_expression< Matrix, Vector, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr matrix_vector_product_expression( Matrix&& m, Vector&& v ) noexcept( Vector::extents_type::static_extent(0) == Matrix::extents_type::static_extent(1) ) :
//       m_(m), v_(v)
//     {
//       if constexpr ( Vector::extents_type::static_extent(0) != Matrix::extents_type::static_extent(1) )
//       {
//         if ( Vector::extents_type::extent(0) != Matrix::extents_type::extent(1) ) LINALG_UNLIKELY
//         {
//           throw length_error( "Vector and Matrix extents are incompatible for matrix-vector multiplication." );
//         }
//       }
//     }
//     constexpr matrix_vector_product_expression& operator = ( const matrix_vector_product_expression& t ) noexcept { this->v_ = t.v_; this->m_ = t.m_; }
//     constexpr matrix_vector_product_expression& operator = ( matrix_vector_product_expression&& t ) noexcept { this->v_ = t.v_; this->m_ = t.m_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<typename Matrix::value_type>() * ::std::declval<typename Vector::value_type>() );
//     using index_type   = ::std::common_type_t< typename Vector::index_type, typename Matrix::index_type >;
//     using size_type    = ::std::common_type_t< typename Vector::size_type, typename Matrix::size_type >;
//     using extents_type = ::std::extents< ::std::common_type_t< typename Vector::extents_type::size_type, typename Matrix::extents_type::size_type >,
//                                          Matrix::extents_type::static_extent(0) >;
//     using rank_type    = ::std::common_type_t< typename Vector::rank_type, typename Matrix::rank_type >;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return 1; }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return extents_type( this->m_.extent(0) ); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->m_.extent(n); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const Vector& first() const noexcept { return this->v_; }
//     [[nodiscard]] constexpr const Matrix& second() const noexcept { return this->m_; }
//     // Matrix Product
//     #if LINALG_USE_BRACKET_OPERATOR
//     [[nodiscard]] constexpr value_type operator[]( index_type index ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<Vector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<Matrix>(), index, ::std::declval<index_type>() ) ) )
//     {
//       return this->access( index );
//     }
//     #endif
//     [[nodiscard]] constexpr value_type operator()( index_type index ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<Vector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<Matrix>(), ::std::declval<index_type>(), index ) ) )
//     {
//       return this->access( index );
//     }
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Access
//     [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION value_type access( const index_type& index ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<Vector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<Matrix>(), ::std::declval<index_type>(), index ) ) )
//     {
//       value_type val { 0 };
//       if constexpr ( Vector::extents_type::static_extent(0) != ::std::dynamic_extent )
//       {
//         for ( auto count = 0; count < this->v_.extent(0); ++count )
//         {
//           val += LINALG_DETAIL::access( this->m_, index, count ) * LINALG_DETAIL::access( this->v_, count );
//         }
//       }
//       else
//       {
//         for ( auto count = 0; count < this->m1_.extent(1); ++count )
//         {
//           val += LINALG_DETAIL::access( this->m_, index, count ) * LINALG_DETAIL::access( this->v_, count );
//         }
//       }
//       return val;
//     }
//     // Data
//     Matrix& m_;
//     Vector& v_;
// };

// #ifdef LINALG_ENABLE_CONCEPTS
// template < LINALG_CONCEPTS::vector_expression FirstVector, LINALG_CONCEPTS::vector_expression SecondVector >
//   requires ( requires ( const typename FirstVector::value_type& v1, const typename SecondVector::value_type& v2 ) { v1 * v2; } )
// #else
// template < class FirstVector, class SecondVector, typename Enable >
// #endif
// class outer_product_expression
// {
//   private:
//     // Alias for self
//     #ifdef LINALG_ENABLE_CONCEPTS
//     using self_type = outer_product_expression< FirstVector, SecondVector >;
//     #else
//     using self_type = outer_product_expression< FirstVector, SecondVector, Enable >;
//     #endif
//   public:
//     // Special member functions
//     constexpr outer_product_expression( FirstVector&& v1, SecondVector&& v2 ) : v1_(v1), v2_(v2) { }
//     constexpr outer_product_expression& operator = ( const outer_product_expression& t ) noexcept { this->v1_ = t.v1_; this->v2_ = t.v2_; }
//     constexpr outer_product_expression& operator = ( outer_product_expression&& t ) noexcept { this->v1_ = t.v1_; this->v2_ = t.v2_; }
//     // Aliases
//     using value_type   = decltype( ::std::declval<typename FirstVector::value_type>() * ::std::declval<typename SecondVector::value_type>() );
//     using index_type   = ::std::common_type_t< typename FirstVector::index_type, typename SecondVector::index_type >;
//     using size_type    = ::std::common_type_t< typename FirstVector::size_type, typename SecondVector::size_type >;
//     using extents_type = ::std::extents< ::std::common_type_t< typename FirstVector::extents_type::size_type, typename SecondVector::extents_type::size_type >,
//                                          FirstVector::extents_type::static_extent(0),
//                                          SecondVector::extents_type::static_extent(0) >;
//     using rank_type    = ::std::common_type_t< typename FirstVector::rank_type, typename SecondVector::rank_type >;
//     // Tensor expression functions
//     [[nodiscard]] static constexpr rank_type rank() noexcept { return 2; }
//     [[nodiscard]] constexpr extents_type extents() noexcept { return extents_type( this->v1_.extent(0), this->v2_.extent(0) ); }
//     [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return ( n == 0 ) ? this->v1_.extent(0) : this->v2_.extent(n-1); }
//     // Binary tensor expression function
//     [[nodiscard]] constexpr const FirstVector& first() const noexcept { return this->v1_; }
//     [[nodiscard]] constexpr const SecondVector& second() const noexcept { return this->v2_; }
//     // Outer Product
//     #if LINALG_USE_BRACKET_OPERATOR
//     [[nodiscard]] constexpr value_type operator[]( index_type index1, index_type index2 ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<FirstVector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<SecondVector>(), ::std::declval<index_type>() ) ) )
//     {
//       return LINALG_ACCESS( this->v1_, index1 ) * LINALG_ACCESS( this->v2_, index2 );
//     }
//     #endif
//     [[nodiscard]] constexpr value_type operator()( index_type index1, index_type index2 ) const
//       noexcept( noexcept( LINALG_DETAIL::access( ::std::declval<FirstVector>(), ::std::declval<index_type>() ) ) &&
//                 noexcept( LINALG_DETAIL::access( ::std::declval<SecondVector>(), ::std::declval<index_type>() ) ) )
//     {
//       return LINALG_ACCESS( this->v1_, index1 ) * LINALG_ACCESS( this->v2_, index2 );
//     }
//     // Implicit conversion
//     [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::rank_dynamic() == 0 ) ?
//                                                       ::std::is_nothrow_constructible_v< fs_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type > :
//                                                       ::std::is_nothrow_constructible_v< dr_tensor< value_type,
//                                                                                                     extents_type,
//                                                                                                     layout_result_t< self_type >,
//                                                                                                     extents_type,
//                                                                                                     typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                                                                                                     accessor_result_t< self_type > >,
//                                                                                          self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
//     {
//       // TODO: Optimizations using tensor traits
//       if constexpr ( extents_type::rank_dynamic() == 0 )
//       {
//         return fs_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           accessor_result_t< self_type > >
//           ( *this );
//       }
//       else
//       {
//         return dr_tensor< value_type,
//                           extents_type,
//                           layout_result_t< self_type >,
//                           extents_type,
//                           typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
//                           accessor_result_t< self_type > >
//           ( *this, allocator_result< self_type >::get_allocator( *this ) );
//       }
//     };
//   private:
//     // Data
//     FirstVector& v1_;
//     SecondVector& v2_;
// };

LINALG_EXPRESSIONS_END // end expressions namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_TENSOR_EXPRESSIONS_HPP
