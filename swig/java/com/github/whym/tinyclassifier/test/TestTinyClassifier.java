package com.github.whym.tinyclassifier.test;
import com.github.whym.tinyclassifier.*;

public class TestTinyClassifier {
  static {
    System.loadLibrary("TinyClassifier");
  }
  public static void main(String[] args) {
    IntPKPerceptron p = new IntPKPerceptron(3);
    System.out.println(""+p.getKernel_order());
  }
}
