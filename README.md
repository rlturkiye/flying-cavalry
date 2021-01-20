# flying-cavalry
Flying Cavalry Project - Ucan Kavalye Projesi 
# Repository - C++ & Makefile

Learn how to create makefile to work with C++ file.

## C ++ 'da Makefile Nasıl Oluşturulur ve Kullanılır

Herhangi bir C ++ projesinde, önemli hedeflerden biri projenin yapımını basitleştirmektir.Bunun için tüm bağımlılıkları ve proje dosyalarını tek bir yerden alır ve tek seferde çalıştırırız, böylece istenen çıktıyı tek bir komutla elde ederiz. Aynı zamanda, proje dosyalarından herhangi biri değiştirildiğinde, tüm projeyi yeniden oluşturma zahmetine girmemize gerek kalmaz, yani projede bir veya iki dosya değiştirildiğinde, yalnızca bu değiştirilen dosyaları yeniden  oluşturur ve ardından yürütmeye(execution) devam ederiz.

Bunlar, C ++ 'da "make" tool ve "makefiles" tarafından ele alınan özelliklerdir. Şimdi, makefile dosyalarının tüm ana yönlerini ve bunların C ++ 'daki uygulamalarını tartışacağız.

### Make Tool

Make bir [UNIX](https://en.wikipedia.org/wiki/Unix) aracıdır ve bir projenin farklı modüllerinden çalıştırılabilir yapıyı basitleştirmek için bir araç olarak kullanılır. Makefile'da hedef girdiler olarak belirtilen çeşitli kurallar vardır. Make tool tüm bu kuralları okur ve buna göre davranır.

**Örneğin**, bir kural herhangi bir bağımlılık belirtiyorsa, bu durumda make tool derleme amaçlı bu bağımlılığı içerecektir. Make komutu makefile'da modüller oluşturmak veya dosyaları temizlemek için kullanılır. 

**Make'in genel sözdizimi şöyledir:**

```
%make target_label             #target_label is a specific target in makefile
```
