#include <gtest/gtest.h>
#include <experimental/linear_algebra.hpp>

namespace
{

  TEST( ADDITION, DR_TENSOR_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor_a{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Add the tensors
    auto add_tensor { tensor_a + tensor_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( add_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( add_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( add_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( add_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( add_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( add_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( add_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( add_tensor, 1, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
    // Check rank
    EXPECT_EQ( ( add_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( add_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( add_tensor.first() ) ), ( ::std::addressof( tensor_a ) ) );
    EXPECT_EQ( ( ::std::addressof( add_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( ADDITION, FS_TENSOR_FS_TENSOR )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Add the tensors
    auto add_tensor { tensor_a + tensor_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( add_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( add_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( add_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( add_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( add_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( add_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( add_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( add_tensor, 1, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
    // Check rank
    EXPECT_EQ( ( add_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( add_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( add_tensor.first() ) ), ( ::std::addressof( tensor_a ) ) );
    EXPECT_EQ( ( ::std::addressof( add_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( ADDITION, DR_TENSOR_FS_TENSOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > tensor_a{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > tensor_b { tensor_a };
    // Add the tensors
    auto add_tensor { tensor_a + tensor_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( add_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( add_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( add_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( add_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( add_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( add_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( add_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( add_tensor, 1, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
    // Check rank
    EXPECT_EQ( ( add_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( add_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( add_tensor.first() ) ), ( ::std::addressof( tensor_a ) ) );
    EXPECT_EQ( ( ::std::addressof( add_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( ADDITION, TENSOR_VIEW_TENSOR_VIEW )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Get subtensors
    auto tensor_a_sub = subtensor( tensor_a, ::std::tuple( 0, 2 ), ::std::tuple( 0, 2 ), 0 );
    auto tensor_b_sub = subtensor( tensor_b, ::std::tuple( 0, 2 ), ::std::tuple( 0, 2 ), 0 );
    // Add the tensors
    auto add_tensor { tensor_a_sub + tensor_b_sub };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( add_tensor, 0, 0 );
    auto val3 = LINALG_DETAIL::access( add_tensor, 0, 1 );
    auto val5 = LINALG_DETAIL::access( add_tensor, 1, 0 );
    auto val7 = LINALG_DETAIL::access( add_tensor, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val7, 14.0 );
    // Check rank
    EXPECT_EQ( ( add_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( add_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( add_tensor.first() ) ), ( ::std::addressof( tensor_a_sub ) ) );
    EXPECT_EQ( ( ::std::addressof( add_tensor.second() ) ), ( ::std::addressof( tensor_b_sub ) ) );
  }

  TEST( ADDITION, TENSOR_VIEW_FS_TENSOR )
  {
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2 > > tensor_b;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_b, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_b, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_b, 1, 1 ) = 4.0;
    // Get subtensor
    auto tensor_a_sub = subtensor( tensor_a, ::std::tuple( 0, 2 ), ::std::tuple( 0, 2 ), 0 );
    // Add the tensors
    auto add_tensor { tensor_a_sub + tensor_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( add_tensor, 0, 0 );
    auto val2 = LINALG_DETAIL::access( add_tensor, 0, 1 );
    auto val3 = LINALG_DETAIL::access( add_tensor, 1, 0 );
    auto val4 = LINALG_DETAIL::access( add_tensor, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 5.0 );
    EXPECT_EQ( val3, 8.0 );
    EXPECT_EQ( val4, 11.0 );
    // Check rank
    EXPECT_EQ( ( add_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( add_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( add_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( add_tensor.first() ) ), ( ::std::addressof( tensor_a_sub ) ) );
    EXPECT_EQ( ( ::std::addressof( add_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( ADDITION_ASSIGNMENT, DR_TENSOR_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor_a{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Add the tensors
    tensor_a += tensor_b;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor_a, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor_a, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor_a, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor_a, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor_a, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor_a, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor_a, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor_a, 1, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( ADDITION_ASSIGNMENT, FS_TENSOR_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    LINALG::dyn_tensor< double, 3 > tensor_b { tensor_a };
    // Add the tensors
    tensor_a += tensor_b;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor_a, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor_a, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor_a, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor_a, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor_a, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor_a, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor_a, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor_a, 1, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( SUBTRACTION, DR_TENSOR_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor_a{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Subtract the tensors
    auto subtract_tensor { tensor_a - tensor_b };
    // Access elements from subtraction expression
    auto val1 = LINALG_DETAIL::access( subtract_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( subtract_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( subtract_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( subtract_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( subtract_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( subtract_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( subtract_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( subtract_tensor, 1, 1, 1 );
    // Check the tensors were subtracted properly
    EXPECT_EQ( val1, 0.0 );
    EXPECT_EQ( val2, 0.0 );
    EXPECT_EQ( val3, 0.0 );
    EXPECT_EQ( val4, 0.0 );
    EXPECT_EQ( val5, 0.0 );
    EXPECT_EQ( val6, 0.0 );
    EXPECT_EQ( val7, 0.0 );
    EXPECT_EQ( val8, 0.0 );
    // Check rank
    EXPECT_EQ( ( subtract_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( subtract_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.first() ) ), ( ::std::addressof( tensor_a ) ) );
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( SUBTRACTION, FS_TENSOR_FS_TENSOR )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Subtract the tensors
    auto subtract_tensor { tensor_a - tensor_b };
    // Access elements from subtraction expression
    auto val1 = LINALG_DETAIL::access( subtract_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( subtract_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( subtract_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( subtract_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( subtract_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( subtract_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( subtract_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( subtract_tensor, 1, 1, 1 );
    // Check the tensors were subtracted properly
    EXPECT_EQ( val1, 0.0 );
    EXPECT_EQ( val2, 0.0 );
    EXPECT_EQ( val3, 0.0 );
    EXPECT_EQ( val4, 0.0 );
    EXPECT_EQ( val5, 0.0 );
    EXPECT_EQ( val6, 0.0 );
    EXPECT_EQ( val7, 0.0 );
    EXPECT_EQ( val8, 0.0 );
    // Check rank
    EXPECT_EQ( ( subtract_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( subtract_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.first() ) ), ( ::std::addressof( tensor_a ) ) );
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( SUBTRACTION, DR_TENSOR_FS_TENSOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > tensor_a{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > tensor_b { tensor_a };
    // Subtract the tensors
    auto subtract_tensor { tensor_a - tensor_b };
    // Access elements from subtraction expression
    auto val1 = LINALG_DETAIL::access( subtract_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( subtract_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( subtract_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( subtract_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( subtract_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( subtract_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( subtract_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( subtract_tensor, 1, 1, 1 );
    // Check the tensors were subtracted properly
    EXPECT_EQ( val1, 0.0 );
    EXPECT_EQ( val2, 0.0 );
    EXPECT_EQ( val3, 0.0 );
    EXPECT_EQ( val4, 0.0 );
    EXPECT_EQ( val5, 0.0 );
    EXPECT_EQ( val6, 0.0 );
    EXPECT_EQ( val7, 0.0 );
    EXPECT_EQ( val8, 0.0 );
    // Check rank
    EXPECT_EQ( ( subtract_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( subtract_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.first() ) ), ( ::std::addressof( tensor_a ) ) );
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( SUBTRACTION, TENSOR_VIEW_TENSOR_VIEW )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Get subtensors
    auto tensor_a_sub = subtensor( tensor_a, ::std::tuple( 0, 2 ), ::std::tuple( 0, 2 ), 0 );
    auto tensor_b_sub = subtensor( tensor_b, ::std::tuple( 0, 2 ), ::std::tuple( 0, 2 ), 0 );
    // Subtract the tensors
    auto subtract_tensor { tensor_a_sub - tensor_b_sub };
    // Access elements from subtraction expression
    auto val1 = LINALG_DETAIL::access( subtract_tensor, 0, 0 );
    auto val3 = LINALG_DETAIL::access( subtract_tensor, 0, 1 );
    auto val5 = LINALG_DETAIL::access( subtract_tensor, 1, 0 );
    auto val7 = LINALG_DETAIL::access( subtract_tensor, 1, 1 );
    // Check the tensors were subtracted properly
    EXPECT_EQ( val1, 0.0 );
    EXPECT_EQ( val3, 0.0 );
    EXPECT_EQ( val5, 0.0 );
    EXPECT_EQ( val7, 0.0 );
    // Check rank
    EXPECT_EQ( ( subtract_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( subtract_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.first() ) ), ( ::std::addressof( tensor_a_sub ) ) );
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.second() ) ), ( ::std::addressof( tensor_b_sub ) ) );
  }

  TEST( SUBTRACTION, TENSOR_VIEW_FS_TENSOR )
  {
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2 > > tensor_b;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_b, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_b, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_b, 1, 1 ) = 4.0;
    // Get subtensor
    auto tensor_a_sub = subtensor( tensor_a, ::std::tuple( 0, 2 ), ::std::tuple( 0, 2 ), 0 );
    // Add the tensors
    auto subtract_tensor { tensor_a_sub - tensor_b };
    // Access elements from subtraction expression
    auto val1 = LINALG_DETAIL::access( subtract_tensor, 0, 0 );
    auto val2 = LINALG_DETAIL::access( subtract_tensor, 0, 1 );
    auto val3 = LINALG_DETAIL::access( subtract_tensor, 1, 0 );
    auto val4 = LINALG_DETAIL::access( subtract_tensor, 1, 1 );
    // Check the tensors were subtracted properly
    EXPECT_EQ( val1, 0.0 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 2.0 );
    EXPECT_EQ( val4, 3.0 );
    // Check rank
    EXPECT_EQ( ( subtract_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( subtract_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( subtract_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.first() ) ), ( ::std::addressof( tensor_a_sub ) ) );
    EXPECT_EQ( ( ::std::addressof( subtract_tensor.second() ) ), ( ::std::addressof( tensor_b ) ) );
  }

  TEST( SUBTRACTION_ASSIGNMENT, DR_TENSOR_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor_a{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    tensor_type tensor_b { tensor_a };
    // Add the tensors
    tensor_a -= tensor_b;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor_a, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor_a, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor_a, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor_a, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor_a, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor_a, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor_a, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor_a, 1, 1, 1 );
    // Check the tensors were subtracted properly
    EXPECT_EQ( val1, 0.0 );
    EXPECT_EQ( val2, 0.0 );
    EXPECT_EQ( val3, 0.0 );
    EXPECT_EQ( val4, 0.0 );
    EXPECT_EQ( val5, 0.0 );
    EXPECT_EQ( val6, 0.0 );
    EXPECT_EQ( val7, 0.0 );
    EXPECT_EQ( val8, 0.0 );
  }

  TEST( SUBTRACTION_ASSIGNMENT, FS_TENSOR_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > tensor_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor_a, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor_a, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor_a, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor_a, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor_a, 1, 1, 1 ) = 8.0;
    // Construct
    LINALG::dyn_tensor< double, 3 > tensor_b { tensor_a };
    // Add the tensors
    tensor_a -= tensor_b;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor_a, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor_a, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor_a, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor_a, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor_a, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor_a, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor_a, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor_a, 1, 1, 1 );
    // Check the tensors were subtracted properly
    EXPECT_EQ( val1, 0.0 );
    EXPECT_EQ( val2, 0.0 );
    EXPECT_EQ( val3, 0.0 );
    EXPECT_EQ( val4, 0.0 );
    EXPECT_EQ( val5, 0.0 );
    EXPECT_EQ( val6, 0.0 );
    EXPECT_EQ( val7, 0.0 );
    EXPECT_EQ( val8, 0.0 );
  }

  TEST( SCALAR_PREPROD, DOUBLE_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Multiply
    auto preprod_tensor { scalar * tensor };
    // Access elements from pre-product
    auto val1 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( preprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( preprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.first() ) ), ( ::std::addressof( scalar ) ) );
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.second() ) ), ( ::std::addressof( tensor ) ) );
  }

  TEST( SCALAR_PREPROD, RVALUE_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Multiply
    auto preprod_tensor { 3 * tensor };
    // Access elements from pre-product
    auto val1 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( preprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( preprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( preprod_tensor.first() ), 3 );
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.second() ) ), ( ::std::addressof( tensor ) ) );
  }

  TEST( SCALAR_PREPROD, FLOAT_DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    float scalar = 3;
    // Multiply
    auto preprod_tensor { scalar * tensor };
    // Access elements from pre-product
    auto val1 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( preprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( preprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.first() ) ), ( ::std::addressof( scalar ) ) );
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.second() ) ), ( ::std::addressof( tensor ) ) );
  }

  TEST( SCALAR_PREPROD, DOUBLE_FS_TENSOR )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Multiply
    auto preprod_tensor { scalar * tensor };
    // Access elements from pre-product
    auto val1 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( preprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( preprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( preprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( preprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( preprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( preprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.first() ) ), ( ::std::addressof( scalar ) ) );
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.second() ) ), ( ::std::addressof( tensor ) ) );
  }

  TEST( SCALAR_PREPROD, DOUBLE_TENSOR_VIEW )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Get subtensor
    auto sub = subtensor( tensor, ::std::tuple(0,2), ::std::tuple(0,2), 0 );
    // Multiply
    auto preprod_tensor { scalar * sub };
    // Access elements from pre-product
    auto val1 = LINALG_DETAIL::access( preprod_tensor, 0, 0 );
    auto val3 = LINALG_DETAIL::access( preprod_tensor, 0, 1 );
    auto val5 = LINALG_DETAIL::access( preprod_tensor, 1, 0 );
    auto val7 = LINALG_DETAIL::access( preprod_tensor, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val7, 21.0 );
    // Check rank
    EXPECT_EQ( ( preprod_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( preprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( preprod_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.first() ) ), ( ::std::addressof( scalar ) ) );
    EXPECT_EQ( ( ::std::addressof( preprod_tensor.second() ) ), ( ::std::addressof( sub ) ) );
  }

  TEST( SCALAR_POSTPROD, DR_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Multiply
    auto postprod_tensor { tensor * scalar };
    // Access elements from post product
    auto val1 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( postprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( postprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
  }

  TEST( SCALAR_POSTPROD, DR_TENSOR_RVALUE )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Multiply
    auto postprod_tensor { tensor * 3 };
    // Access elements from post product
    auto val1 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( postprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( postprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( postprod_tensor.second() ), 3 );
  }

  TEST( SCALAR_POSTPROD, DR_TENSOR_FLOAT )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    float scalar = 3;
    // Multiply
    auto postprod_tensor { tensor * scalar };
    // Access elements from post product
    auto val1 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( postprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( postprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_POSTPROD, FS_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Multiply
    auto postprod_tensor { tensor * scalar };
    // Access elements from post product
    auto val1 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( postprod_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( postprod_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( postprod_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( postprod_tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
    // Check rank
    EXPECT_EQ( ( postprod_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( postprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_POSTPROD, TENSOR_VIEW_DOUBLE )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Get subtensor
    auto sub = subtensor( tensor, ::std::tuple(0,2), ::std::tuple(0,2), 0 );
    // Multiply
    auto postprod_tensor { sub * scalar };
    // Access elements from post product
    auto val1 = LINALG_DETAIL::access( postprod_tensor, 0, 0 );
    auto val3 = LINALG_DETAIL::access( postprod_tensor, 0, 1 );
    auto val5 = LINALG_DETAIL::access( postprod_tensor, 1, 0 );
    auto val7 = LINALG_DETAIL::access( postprod_tensor, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val7, 21.0 );
    // Check rank
    EXPECT_EQ( ( postprod_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( postprod_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( postprod_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.first() ) ), ( ::std::addressof( sub ) ) );
    EXPECT_EQ( ( ::std::addressof( postprod_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_POSTPROD_ASSIGNMENT, DR_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Multiply
    tensor *= scalar;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
  }

  TEST( SCALAR_POSTPROD_ASSIGNMENT, FS_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Multiply
    tensor *= scalar;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor, 1, 1, 1 );
    // Check the tensor was multiplied properly
    EXPECT_EQ( val1, 3.0 );
    EXPECT_EQ( val2, 6.0 );
    EXPECT_EQ( val3, 9.0 );
    EXPECT_EQ( val4, 12.0 );
    EXPECT_EQ( val5, 15.0 );
    EXPECT_EQ( val6, 18.0 );
    EXPECT_EQ( val7, 21.0 );
    EXPECT_EQ( val8, 24.0 );
  }

  TEST( SCALAR_DIVISION, DR_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Divide
    auto division_tensor { tensor / scalar };
    // Access elements from division expression
    auto val1 = LINALG_DETAIL::access( division_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( division_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( division_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( division_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( division_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( division_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( division_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( division_tensor, 1, 1, 1 );
    // Check the tensor was divided properly
    EXPECT_EQ( val1, 1.0 / 3.0 );
    EXPECT_EQ( val2, 2.0 / 3.0 );
    EXPECT_EQ( val3, 1.0 );
    EXPECT_EQ( val4, 4.0 / 3.0 );
    EXPECT_EQ( val5, 5.0 / 3.0 );
    EXPECT_EQ( val6, 2.0 );
    EXPECT_EQ( val7, 7.0 / 3.0 );
    EXPECT_EQ( val8, 8.0 / 3.0 );
    // Check rank
    EXPECT_EQ( ( division_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( division_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( division_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
    EXPECT_EQ( ( ::std::addressof( division_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
  }

  TEST( SCALAR_DIVISION, DR_TENSOR_RVALUE )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Division
    auto division_tensor { tensor / 3 };
    // Access elements from division expression
    auto val1 = LINALG_DETAIL::access( division_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( division_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( division_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( division_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( division_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( division_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( division_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( division_tensor, 1, 1, 1 );
    // Check the tensor was divided properly
    EXPECT_EQ( val1, 1.0 / 3.0 );
    EXPECT_EQ( val2, 2.0 / 3.0 );
    EXPECT_EQ( val3, 1.0 );
    EXPECT_EQ( val4, 4.0 / 3.0 );
    EXPECT_EQ( val5, 5.0 / 3.0 );
    EXPECT_EQ( val6, 2.0 );
    EXPECT_EQ( val7, 7.0 / 3.0 );
    EXPECT_EQ( val8, 8.0 / 3.0 );
    // Check rank
    EXPECT_EQ( ( division_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( division_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( division_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( division_tensor.second() ), 3 );
  }

  TEST( SCALAR_DIVISION, DR_TENSOR_FLOAT )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    float scalar = 3;
    // Division
    auto division_tensor { tensor / scalar };
    // Access elements from division expression
    auto val1 = LINALG_DETAIL::access( division_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( division_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( division_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( division_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( division_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( division_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( division_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( division_tensor, 1, 1, 1 );
    // Check the tensor was divided properly
    EXPECT_EQ( val1, 1.0 / 3.0 );
    EXPECT_EQ( val2, 2.0 / 3.0 );
    EXPECT_EQ( val3, 1.0 );
    EXPECT_EQ( val4, 4.0 / 3.0 );
    EXPECT_EQ( val5, 5.0 / 3.0 );
    EXPECT_EQ( val6, 2.0 );
    EXPECT_EQ( val7, 7.0 / 3.0 );
    EXPECT_EQ( val8, 8.0 / 3.0 );
    // Check rank
    EXPECT_EQ( ( division_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( division_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( division_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( ::std::addressof( division_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_DIVISION, FS_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Division
    auto division_tensor { tensor / scalar };
    // Access elements from division expression
    auto val1 = LINALG_DETAIL::access( division_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( division_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( division_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( division_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( division_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( division_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( division_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( division_tensor, 1, 1, 1 );
    // Check the tensor was divided properly
    EXPECT_EQ( val1, 1.0 / 3.0 );
    EXPECT_EQ( val2, 2.0 / 3.0 );
    EXPECT_EQ( val3, 1.0 );
    EXPECT_EQ( val4, 4.0 / 3.0 );
    EXPECT_EQ( val5, 5.0 / 3.0 );
    EXPECT_EQ( val6, 2.0 );
    EXPECT_EQ( val7, 7.0 / 3.0 );
    EXPECT_EQ( val8, 8.0 / 3.0 );
    // Check rank
    EXPECT_EQ( ( division_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( division_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( division_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( ::std::addressof( division_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_DIVISION, TENSOR_VIEW_DOUBLE )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Get subtensor
    auto sub = subtensor( tensor, ::std::tuple(0,2), ::std::tuple(0,2), 0 );
    // Division
    auto division_tensor { sub / scalar };
    // Access elements from division expression
    auto val1 = LINALG_DETAIL::access( division_tensor, 0, 0 );
    auto val3 = LINALG_DETAIL::access( division_tensor, 0, 1 );
    auto val5 = LINALG_DETAIL::access( division_tensor, 1, 0 );
    auto val7 = LINALG_DETAIL::access( division_tensor, 1, 1 );
    // Check the tensor was divided properly
    EXPECT_EQ( val1, 1.0 / 3.0 );
    EXPECT_EQ( val3, 1.0 );
    EXPECT_EQ( val5, 5.0 / 3.0 );
    EXPECT_EQ( val7, 7.0 / 3.0 );
    // Check rank
    EXPECT_EQ( ( division_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( division_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( division_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( division_tensor.first() ) ), ( ::std::addressof( sub ) ) );
    EXPECT_EQ( ( ::std::addressof( division_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_DIVISION_ASSIGNMENT, DR_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Divide
    tensor /= scalar;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor, 1, 1, 1 );
    // Check the tensor was divided properly
    EXPECT_EQ( val1, 1.0 / 3.0 );
    EXPECT_EQ( val2, 2.0 / 3.0 );
    EXPECT_EQ( val3, 1.0 );
    EXPECT_EQ( val4, 4.0 / 3.0 );
    EXPECT_EQ( val5, 5.0 / 3.0 );
    EXPECT_EQ( val6, 2.0 );
    EXPECT_EQ( val7, 7.0 / 3.0 );
    EXPECT_EQ( val8, 8.0 / 3.0 );
  }

  TEST( SCALAR_DIVISION_ASSIGNMENT, FS_TENSOR_DOUBLE )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    double scalar = 3;
    // Division
    tensor /= scalar;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor, 1, 1, 1 );
    // Check the tensor was divided properly
    EXPECT_EQ( val1, 1.0 / 3.0 );
    EXPECT_EQ( val2, 2.0 / 3.0 );
    EXPECT_EQ( val3, 1.0 );
    EXPECT_EQ( val4, 4.0 / 3.0 );
    EXPECT_EQ( val5, 5.0 / 3.0 );
    EXPECT_EQ( val6, 2.0 );
    EXPECT_EQ( val7, 7.0 / 3.0 );
    EXPECT_EQ( val8, 8.0 / 3.0 );
  }

  TEST( SCALAR_MODULO, DR_TENSOR_INT )
  {
    using tensor_type = LINALG::dyn_tensor< int, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8;
    // Scalar
    int scalar = 3;
    // Modulo
    auto modulo_tensor { tensor % scalar };
    // Access elements from modulo expression
    auto val1 = LINALG_DETAIL::access( modulo_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( modulo_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( modulo_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( modulo_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( modulo_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( modulo_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( modulo_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( modulo_tensor, 1, 1, 1 );
    // Check the tensor was modulo properly
    EXPECT_EQ( val1, 1 );
    EXPECT_EQ( val2, 2 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 1  );
    EXPECT_EQ( val5, 2 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 1 );
    EXPECT_EQ( val8, 2 );
    // Check rank
    EXPECT_EQ( ( modulo_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( modulo_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( modulo_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
    EXPECT_EQ( ( ::std::addressof( modulo_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
  }

  TEST( SCALAR_MODULO, DR_TENSOR_RVALUE )
  {
    using tensor_type = LINALG::dyn_tensor< int, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Modulo
    auto modulo_tensor { tensor % 3 };
    // Access elements from modulo expression
    auto val1 = LINALG_DETAIL::access( modulo_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( modulo_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( modulo_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( modulo_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( modulo_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( modulo_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( modulo_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( modulo_tensor, 1, 1, 1 );
    // Check the tensor was modulo properly
    EXPECT_EQ( val1, 1 );
    EXPECT_EQ( val2, 2 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 1 );
    EXPECT_EQ( val5, 2 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 1 );
    EXPECT_EQ( val8, 2 );
    // Check rank
    EXPECT_EQ( ( modulo_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( modulo_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( modulo_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( modulo_tensor.second() ), 3 );
  }

  TEST( SCALAR_MODULO, FS_TENSOR_INT )
  {
    using tensor_type = LINALG::fs_tensor< int, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Scalar
    int scalar = 3;
    // Modulo
    auto modulo_tensor { tensor % scalar };
    // Access elements from division expression
    auto val1 = LINALG_DETAIL::access( modulo_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( modulo_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( modulo_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( modulo_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( modulo_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( modulo_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( modulo_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( modulo_tensor, 1, 1, 1 );
    // Check the tensor was modulo properly
    EXPECT_EQ( val1, 1 );
    EXPECT_EQ( val2, 2 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 1 );
    EXPECT_EQ( val5, 2 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 1 );
    EXPECT_EQ( val8, 2 );
    // Check rank
    EXPECT_EQ( ( modulo_tensor.rank() ), 3 );
    // Check extents
    EXPECT_EQ( ( modulo_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extent(2) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(1) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(2) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( modulo_tensor.first() ) ), ( ::std::addressof( tensor ) ) );
    EXPECT_EQ( ( ::std::addressof( modulo_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_MODULO, TENSOR_VIEW_INT )
  {
    using tensor_type = LINALG::fs_tensor< int, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8;
    // Scalar
    int scalar = 3;
    // Get subtensor
    auto sub = subtensor( tensor, ::std::tuple(0,2), ::std::tuple(0,2), 0 );
    // Modulo
    auto modulo_tensor { sub % scalar };
    // Access elements from modulo expression
    auto val1 = LINALG_DETAIL::access( modulo_tensor, 0, 0 );
    auto val3 = LINALG_DETAIL::access( modulo_tensor, 0, 1 );
    auto val5 = LINALG_DETAIL::access( modulo_tensor, 1, 0 );
    auto val7 = LINALG_DETAIL::access( modulo_tensor, 1, 1 );
    // Check the tensor was modulo properly
    EXPECT_EQ( val1, 1 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val5, 2 );
    EXPECT_EQ( val7, 1 );
    // Check rank
    EXPECT_EQ( ( modulo_tensor.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( modulo_tensor.extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extent(1) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(0) ), 2 );
    EXPECT_EQ( ( modulo_tensor.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( modulo_tensor.first() ) ), ( ::std::addressof( sub ) ) );
    EXPECT_EQ( ( ::std::addressof( modulo_tensor.second() ) ), ( ::std::addressof( scalar ) ) );
  }

  TEST( SCALAR_MODULO_ASSIGNMENT, DR_TENSOR_INT )
  {
    using tensor_type = LINALG::dyn_tensor< int, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8;
    // Scalar
    int scalar = 3;
    // Modulo
    tensor %= scalar;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor, 1, 1, 1 );
    // Check the tensor was modulo properly
    EXPECT_EQ( val1, 1 );
    EXPECT_EQ( val2, 2 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 1 );
    EXPECT_EQ( val5, 2 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 1 );
    EXPECT_EQ( val8, 2 );
  }

  TEST( SCALAR_MODULO_ASSIGNMENT, FS_TENSOR_INT )
  {
    using tensor_type = LINALG::fs_tensor< int, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8;
    // Scalar
    int scalar = 3;
    // Modulo
    tensor %= scalar;
    // Access elements from tensor
    auto val1 = LINALG_DETAIL::access( tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( tensor, 1, 1, 1 );
    // Check the tensor was modulo properly
    EXPECT_EQ( val1, 1 );
    EXPECT_EQ( val2, 2 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 1 );
    EXPECT_EQ( val5, 2 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 1 );
    EXPECT_EQ( val8, 2 );
  }

  TEST( MATRIC_PRODUCT, DR_MATRIX_DR_MATRIX )
  {
    using matrix_type = LINALG::dyn_matrix< double >;
    // Construct
    matrix_type matrix_a{ ::std::extents< ::std::size_t, 2, 3 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_a, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_a, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_a, 0, 2 ) = 3.0;
    LINALG_DETAIL::access( matrix_a, 1, 0 ) = 4.0;
    LINALG_DETAIL::access( matrix_a, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_a, 1, 2 ) = 6.0;
    // Construct
    matrix_type matrix_b { ::std::extents< ::std::size_t, 3, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_b, 1, 0 ) = 2.0;
    LINALG_DETAIL::access( matrix_b, 2, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_b, 0, 1 ) = 4.0;
    LINALG_DETAIL::access( matrix_b, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_b, 2, 1 ) = 6.0;
    // Multiply the matrices
    auto prod_matrix { matrix_a * matrix_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( prod_matrix, 0, 0 );
    auto val2 = LINALG_DETAIL::access( prod_matrix, 0, 1 );
    auto val3 = LINALG_DETAIL::access( prod_matrix, 1, 0 );
    auto val4 = LINALG_DETAIL::access( prod_matrix, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
    // Check rank
    EXPECT_EQ( ( prod_matrix.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( prod_matrix.extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extent(1) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( prod_matrix.first() ) ), ( ::std::addressof( matrix_a ) ) );
    EXPECT_EQ( ( ::std::addressof( prod_matrix.second() ) ), ( ::std::addressof( matrix_b ) ) );
  }

  TEST( MATRIX_PRODUCT, FS_MATRIX_FS_MATRIX )
  {
    // Construct
    LINALG::fs_matrix< double, 2, 3 > matrix_a;
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_a, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_a, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_a, 0, 2 ) = 3.0;
    LINALG_DETAIL::access( matrix_a, 1, 0 ) = 4.0;
    LINALG_DETAIL::access( matrix_a, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_a, 1, 2 ) = 6.0;
    // Construct
    LINALG::fs_matrix< double, 3, 2 > matrix_b;
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_b, 1, 0 ) = 2.0;
    LINALG_DETAIL::access( matrix_b, 2, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_b, 0, 1 ) = 4.0;
    LINALG_DETAIL::access( matrix_b, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_b, 2, 1 ) = 6.0;
    // Multiply the matrices
    auto prod_matrix { matrix_a * matrix_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( prod_matrix, 0, 0 );
    auto val2 = LINALG_DETAIL::access( prod_matrix, 0, 1 );
    auto val3 = LINALG_DETAIL::access( prod_matrix, 1, 0 );
    auto val4 = LINALG_DETAIL::access( prod_matrix, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
    // Check rank
    EXPECT_EQ( ( prod_matrix.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( prod_matrix.extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extent(1) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( prod_matrix.first() ) ), ( ::std::addressof( matrix_a ) ) );
    EXPECT_EQ( ( ::std::addressof( prod_matrix.second() ) ), ( ::std::addressof( matrix_b ) ) );
  }

  TEST( MATRIX_PRODUCT, DR_MATRIX_FS_MATRIX )
  {
    // Construct
    LINALG::dyn_matrix< double > matrix_a { ::std::extents< ::std::size_t, 2, 3 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_a, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_a, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_a, 0, 2 ) = 3.0;
    LINALG_DETAIL::access( matrix_a, 1, 0 ) = 4.0;
    LINALG_DETAIL::access( matrix_a, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_a, 1, 2 ) = 6.0;
    // Construct
    LINALG::fs_matrix< double, 3, 2 > matrix_b;
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_b, 1, 0 ) = 2.0;
    LINALG_DETAIL::access( matrix_b, 2, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_b, 0, 1 ) = 4.0;
    LINALG_DETAIL::access( matrix_b, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_b, 2, 1 ) = 6.0;
    // Multiply the matrices
    auto prod_matrix { matrix_a * matrix_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( prod_matrix, 0, 0 );
    auto val2 = LINALG_DETAIL::access( prod_matrix, 0, 1 );
    auto val3 = LINALG_DETAIL::access( prod_matrix, 1, 0 );
    auto val4 = LINALG_DETAIL::access( prod_matrix, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
    // Check rank
    EXPECT_EQ( ( prod_matrix.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( prod_matrix.extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extent(1) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( prod_matrix.first() ) ), ( ::std::addressof( matrix_a ) ) );
    EXPECT_EQ( ( ::std::addressof( prod_matrix.second() ) ), ( ::std::addressof( matrix_b ) ) );
  }

  TEST( MATRIX_PRODUCT, MATRIX_VIEW_MATRIX_VIEW )
  {
    // Construct
    LINALG::dyn_matrix< double > matrix_a { ::std::extents< ::std::size_t, 3, 3 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_a, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_a, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_a, 0, 2 ) = 3.0;
    LINALG_DETAIL::access( matrix_a, 1, 0 ) = 4.0;
    LINALG_DETAIL::access( matrix_a, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_a, 1, 2 ) = 6.0;
    LINALG_DETAIL::access( matrix_a, 2, 0 ) = 7.0;
    LINALG_DETAIL::access( matrix_a, 2, 1 ) = 8.0;
    LINALG_DETAIL::access( matrix_a, 2, 2 ) = 9.0;
    // Construct
    LINALG::fs_matrix< double, 3, 3 > matrix_b;
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_b, 1, 0 ) = 2.0;
    LINALG_DETAIL::access( matrix_b, 2, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_b, 0, 1 ) = 4.0;
    LINALG_DETAIL::access( matrix_b, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_b, 2, 1 ) = 6.0;
    LINALG_DETAIL::access( matrix_b, 0, 2 ) = 7.0;
    LINALG_DETAIL::access( matrix_b, 1, 2 ) = 8.0;
    LINALG_DETAIL::access( matrix_b, 2, 2 ) = 9.0;
    // Get submatrix views
    auto submatrix_a = submatrix( matrix_a, ::std::tuple(0,2), ::std::tuple(0,3) );
    auto submatrix_b = submatrix( matrix_b, ::std::tuple(0,3), ::std::tuple(0,2) );
    // Multiply the matrices
    auto prod_matrix { submatrix_a * submatrix_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( prod_matrix, 0, 0 );
    auto val2 = LINALG_DETAIL::access( prod_matrix, 0, 1 );
    auto val3 = LINALG_DETAIL::access( prod_matrix, 1, 0 );
    auto val4 = LINALG_DETAIL::access( prod_matrix, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
    // Check rank
    EXPECT_EQ( ( prod_matrix.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( prod_matrix.extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extent(1) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( prod_matrix.first() ) ), ( ::std::addressof( submatrix_a ) ) );
    EXPECT_EQ( ( ::std::addressof( prod_matrix.second() ) ), ( ::std::addressof( submatrix_b ) ) );
  }

  TEST( MATRIX_PRODUCT, TENSOR_VIEW_FS_TENSOR )
  {
    // Construct
    LINALG::dyn_matrix< double > matrix_a { ::std::extents< ::std::size_t, 3, 3 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_a, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_a, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_a, 0, 2 ) = 3.0;
    LINALG_DETAIL::access( matrix_a, 1, 0 ) = 4.0;
    LINALG_DETAIL::access( matrix_a, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_a, 1, 2 ) = 6.0;
    LINALG_DETAIL::access( matrix_a, 2, 0 ) = 7.0;
    LINALG_DETAIL::access( matrix_a, 2, 1 ) = 8.0;
    LINALG_DETAIL::access( matrix_a, 2, 2 ) = 9.0;
    // Get submatrix view
    auto submatrix_a = submatrix( matrix_a, ::std::tuple(0,2), ::std::tuple(0,3) );
    // Construct
    LINALG::fs_matrix< double, 3, 2 > matrix_b;
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_b, 1, 0 ) = 2.0;
    LINALG_DETAIL::access( matrix_b, 2, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_b, 0, 1 ) = 4.0;
    LINALG_DETAIL::access( matrix_b, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_b, 2, 1 ) = 6.0;
    // Multiply the matrices
    auto prod_matrix { submatrix_a * matrix_b };
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( prod_matrix, 0, 0 );
    auto val2 = LINALG_DETAIL::access( prod_matrix, 0, 1 );
    auto val3 = LINALG_DETAIL::access( prod_matrix, 1, 0 );
    auto val4 = LINALG_DETAIL::access( prod_matrix, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
    // Check rank
    EXPECT_EQ( ( prod_matrix.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( prod_matrix.extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extent(1) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(0) ), 2 );
    EXPECT_EQ( ( prod_matrix.extents().extent(1) ), 2 );
    // Check underlying types
    EXPECT_EQ( ( ::std::addressof( prod_matrix.first() ) ), ( ::std::addressof( submatrix_a ) ) );
    EXPECT_EQ( ( ::std::addressof( prod_matrix.second() ) ), ( ::std::addressof( matrix_b ) ) );
  }

  TEST( MATRIX_PRODUCT_ASSIGNMENT, DR_TENSOR_DR_TENSOR )
  {
    using matrix_type = LINALG::dyn_matrix< double >;
    // Construct
    matrix_type matrix_a{ ::std::extents< ::std::size_t, 2, 3 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_a, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_a, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_a, 0, 2 ) = 3.0;
    LINALG_DETAIL::access( matrix_a, 1, 0 ) = 4.0;
    LINALG_DETAIL::access( matrix_a, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_a, 1, 2 ) = 6.0;
    // Construct
    matrix_type matrix_b { ::std::extents< ::std::size_t, 3, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_b, 1, 0 ) = 2.0;
    LINALG_DETAIL::access( matrix_b, 2, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_b, 0, 1 ) = 4.0;
    LINALG_DETAIL::access( matrix_b, 1, 1 ) = 5.0;
    LINALG_DETAIL::access( matrix_b, 2, 1 ) = 6.0;
    // Multiply the matrices
    matrix_a *= matrix_b;
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( matrix_a, 0, 0 );
    auto val2 = LINALG_DETAIL::access( matrix_a, 0, 1 );
    auto val3 = LINALG_DETAIL::access( matrix_a, 1, 0 );
    auto val4 = LINALG_DETAIL::access( matrix_a, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
    // Check rank
    EXPECT_EQ( ( matrix_a.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( matrix_a.extent(0) ), 2 );
    EXPECT_EQ( ( matrix_a.extent(1) ), 2 );
    EXPECT_EQ( ( matrix_a.extents().extent(0) ), 2 );
    EXPECT_EQ( ( matrix_a.extents().extent(1) ), 2 );
  }

  TEST( MATRIX_PRODUCT_ASSIGNMENT, FS_TENSOR_DR_TENSOR )
  {
    // Construct
    LINALG::fs_matrix< double, 2, 2 > matrix_a { };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_a, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_a, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_a, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_a, 1, 1 ) = 4.0;
    // Construct
    LINALG::dyn_matrix< double > matrix_b { ::std::extents< ::std::size_t, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( matrix_b, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( matrix_b, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( matrix_b, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( matrix_b, 1, 1 ) = 4.0;
    // Multiply the matrices
    matrix_a *= matrix_b;
    // Access elements from addition expression
    auto val1 = LINALG_DETAIL::access( matrix_a, 0, 0 );
    auto val2 = LINALG_DETAIL::access( matrix_a, 0, 1 );
    auto val3 = LINALG_DETAIL::access( matrix_a, 1, 0 );
    auto val4 = LINALG_DETAIL::access( matrix_a, 1, 1 );
    // Check the tensors were added properly
    EXPECT_EQ( val1, 9.0 );
    EXPECT_EQ( val2, 12.0 );
    EXPECT_EQ( val3, 24.0 );
    EXPECT_EQ( val4, 33.0 );
    // Check rank
    EXPECT_EQ( ( matrix_a.rank() ), 2 );
    // Check extents
    EXPECT_EQ( ( matrix_a.extent(0) ), 2 );
    EXPECT_EQ( ( matrix_a.extent(1) ), 2 );
    EXPECT_EQ( ( matrix_a.extents().extent(0) ), 2 );
    EXPECT_EQ( ( matrix_a.extents().extent(1) ), 2 );
  }

}