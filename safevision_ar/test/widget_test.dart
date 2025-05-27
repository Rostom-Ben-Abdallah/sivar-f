// test/widget_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:safevision_ar/main.dart';

void main() {
  testWidgets('App loads and shows title', (WidgetTester tester) async {
    // Build the app
    await tester.pumpWidget(const SafeVisionApp());
    await tester.pumpAndSettle();

    // Verify that the AppBar title is displayed
    expect(find.text('SafeVision AR'), findsOneWidget);
  });
}
