(defproject oa4j-dl "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/data.json "0.2.6"]
                 [at.rocworks.oa4j/winccoa-java "1.0-SNAPSHOT"]]
  :repositories {"sonartype snapshots" "https://oss.sonatype.org/content/repositories/snapshots"}
  :jvm-opts ["-Djava.library.path=/opt/WinCC_OA/current/bin"]
  :aot  [dl.core]
  :main dl.core)