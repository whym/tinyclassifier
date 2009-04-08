import com.github.whym.*;

public class test {
  static {
    System.loadLibrary("TinyClassifier");
  }
  public static void main(String[] args) {
    IntPerceptron p = new IntPerceptron(3);
    System.out.println(""+p.getKernel_order());
  }
}
