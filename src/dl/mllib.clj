(ns dl.mllib)

;------------------------------------------------------------------------------------------
(defn matrix [xs]
  "create a matrix from an array [1 2 3] => [[1] [2] [3]]"
  (mapv #(vector %) xs))

(defn transpose [matrix]
  (println "TRANSPOSE" matrix)
  (apply mapv vector matrix))

(defn product [a b]
  (doseq [x a] (assert (= (count x) (count b)) "The number of columns in first matrix should be equal to the number of rows in the second"))
  (doseq [x b] (assert (= (count (first b)) (count x)) "Number of columns in second matrix is not consistent"))
  (let [nested-for (fn [f x y] (mapv (fn [a] (mapv (fn [b] (f a b)) y)) x))]
    (nested-for (fn [x y] (reduce + (mapv * x y))) a (transpose b))))

(defn output-errors-calculation [outputs targets]
  "calculate the errors for the output layer (target value - actual value)"
  ; outputs [[0.02] [0.027]]
  ; targets [0 1]
  (let [outputs (flatten outputs) targets (flatten targets)]
    (mapv - targets outputs)))

(defn hidden-errors-calculation [activations errors weights dactivation-fn]
  "calculate errors for the hidden layer based on the output-errors and weights"
  ; activations [
  ; errors [0.8 0.5]
  ; weights [[0.15 0.02 0.01] [0.16 0.03 0.02]]
  (println "{-- hidden-errors-calculation ---")
  (println "activations =" activations)
  (println "errors      =" errors)
  (println "weights     =" weights)
  (println "--- hidden-errors-calculation --}")
  (let [activations (flatten activations)]
    (mapv *
         (mapv dactivation-fn activations)
         (flatten (product (matrix errors) (transpose weights))))))

(defn weights-calculation [activations errors weights rate]
  "update/adjust the weights of a hidden layer according to the errors"
  ; activations [[0.02] [0.02]]
  ; errors [0.8 0.5]
  ; weights [[0.15 0.02 0.01] [0.16 0.03 0.02]] e.g. 3 neurons follow
  (let [layer-gradients (mapv * errors (flatten activations))] ; (0.018520152042568426 0.013804030750041782)
    (mapv
      (fn [ws g]
        (mapv (fn [w] (+ w (* g rate))) ws))
      weights layer-gradients)))