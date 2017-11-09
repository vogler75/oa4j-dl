(ns dl.core
  (:gen-class)
  (:require [dl.oa4j :as scada]
            [dl.mllib :as ml]
            [clojure.data.json :as json])
  (:import (at.rocworks.oa4j.base JDpIdValueList JDpVCItem)
           (at.rocworks.oa4j.var DynVar)))

;;-----------------------------------------------------------------------------------------------------------------------
(defn activation-fn-tanh [x] "tanh activation" (Math/tanh x))
(defn dactivation-fn-tanh [y] "tanh derivative " (- 1.0 (* y y)))

(defn activation-fn-sigmoid [x] "sigmoid activation" (/ 1 (+ 1 (Math/exp (* -1 x)))))
(defn dactivation-fn-sigmoid [x] "sigmoid derivative" (/ (Math/exp x) (Math/pow (+ 1 (Math/exp x)) 2)))

;;-----------------------------------------------------------------------------------------------------------------------
(def nn-rate (atom 0.0))
(def nn-layers (atom (list)))
(def nn-weights (atom (list)))
(def nn-outputs (atom (list)))

(def nn-activation-fn (atom activation-fn-sigmoid))
(def nn-dactivation-fn (atom dactivation-fn-sigmoid))

;-----------------------------------------------------------------------------------------------------------------------
(defn scada-get-weights []
  (reset! nn-weights (json/read-str (scada/dpGet :Network.Network.Weights))))

(defn scada-set-weights []
  (scada/dpSet :Network.Network.Weights (json/write-str @nn-weights)))

(defn scada-set-output []
  (scada/dpSet :Network.Data.Output (DynVar. (to-array (first @nn-outputs)))))

;-----------------------------------------------------------------------------------------------------------------------
; layers [2 3 2]

; weights layer 1 (input to hidden 2=>3)
;[[0.12 0.01]
; [0.20 0.02]
; [0.13 0.03]]

; weights layer 2 (hidden to output 3=>2)
;[[0.15 0.02 0.01]
; [0.16 0.03 0.02]]

(defn init-network [layers]
  (println "init-network" layers)
  (reset! nn-layers layers)
  (reset! nn-weights (mapv ml/init-weight-matrix layers (rest layers))))

;-----------------------------------------------------------------------------------------------------------------------
(defn reset-callback [[layers]]
  (init-network layers))

(defn set-activation-fn-callback [[nr]]
  (case nr
    0 (do
        (reset! nn-activation-fn activation-fn-tanh)
        (reset! nn-dactivation-fn dactivation-fn-tanh))
    1 (do
        (reset! nn-activation-fn activation-fn-sigmoid)
        (reset! nn-dactivation-fn dactivation-fn-sigmoid))))

(defn set-rate-callback [[rate]]
  (reset! nn-rate rate))

(defn save-network-callback [[trigger]]
  (scada-set-weights))

(defn feed-callback [[input target train]]
  (if train
    (reset! nn-weights (ml/train-network input @nn-weights target @nn-rate @nn-activation-fn @nn-dactivation-fn))
    (reset! nn-outputs (ml/feed-network (ml/matrix input) @nn-weights @nn-activation-fn)))
  (scada-set-output))

;-----------------------------------------------------------------------------------------------------------------------
(def xor-data [{:i [0 0] :t [0]}
               {:i [0 1] :t [1]}
               {:i [1 0] :t [1]}
               {:i [1 1] :t [0]}])

(defn xor-test []
  (doall
    (map
      #(println (:i %) "=>" (ml/feed-network (ml/matrix (:i %)) @nn-weights @nn-activation-fn))
                xor-data)))

;-----------------------------------------------------------------------------------------------------------------------
(defn -main [& args]
  (comment
    (scada/startup args)

    (scada-get-weights)
    (println "weights" @nn-weights)

    (scada/dpConnect [:Network.Config.Layers] reset-callback)
    (scada/dpConnect [:Network.Config.ActivationFn] set-activation-fn-callback true)

    (scada/dpConnect [:Network.Control.Rate] set-rate-callback true)

    (scada/dpConnect [:Network.Data.Input
                      :Network.Data.Target
                      :Network.Control.Train] feed-callback)

    (scada/dpConnect [:Network.Control.Save] save-network-callback)
    )

  ; xor-test
  (init-network [2 2 1])
  (println "inital network: " @nn-weights)
  (dotimes [_ 20000]
    (doseq [x xor-data]
      (reset! nn-weights (ml/train-network (:i x) @nn-weights (:t x) 0.1 @nn-activation-fn @nn-dactivation-fn))))

  (xor-test)

  ; already trained network for XOR, just to compare the output with another NN program
  (reset! nn-weights '((( 5.986938319255103 -3.9050378273319755  -3.8927622364550345 )
                         (1.99893455091257 -5.049468802938935   -5.0079142506851575   ))
                        ((-7.04504038559793 14.718764645089621  -14.907682119185624   ))))
  (println "inital network: " @nn-weights)
  (let [x (first xor-data)
        n (ml/train-network (:i x) @nn-weights (:t x) 0.1 @nn-activation-fn @nn-dactivation-fn)]
    (println "trained network: " n))

  (println "ready")
)


