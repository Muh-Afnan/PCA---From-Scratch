import unittest


class VisualTestResult(unittest.TextTestResult):

    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"✅ SUCCESS: {test}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"❌ FAILURE: {test}")

    def addError(self, test, err):
        super().addError(test, err)
        print(f"⚠️ ERROR:   {test}")

    def stopTestRun(self):
        super().stopTestRun()

        total = self.testsRun
        failures = len(self.failures)
        errors = len(self.errors)
        success = total - failures - errors

        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Total   : {total}")
        print(f"Success : {success} ✅")
        print(f"Failures: {failures} ❌")
        print(f"Errors  : {errors} ⚠️")
        print("=" * 50)


class VisualTestRunner(unittest.TextTestRunner):
    resultclass = VisualTestResult


if __name__ == "__main__":
    loader = unittest.TestLoader()

    # discover tests automatically
    suite = loader.discover("tests")

    runner = VisualTestRunner(verbosity=0)
    runner.run(suite)
