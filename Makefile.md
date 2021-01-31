# Repository - C++ & Makefile

## C ++ 'da Makefile Nasıl Oluşturulur ve Kullanılır

Herhangi bir C ++ projesinde, önemli hedeflerden biri projenin yapımını basitleştirmektir.Bunun için tüm bağımlılıkları ve proje dosyalarını tek bir yerden alır ve tek seferde çalıştırırız, böylece istenen çıktıyı tek bir komutla elde ederiz. Aynı zamanda, proje dosyalarından herhangi biri değiştirildiğinde, tüm projeyi yeniden oluşturma zahmetine girmemize gerek kalmaz, yani projede bir veya iki dosya değiştirildiğinde, yalnızca bu değiştirilen dosyaları yeniden  oluşturur ve ardından yürütmeye(execution) devam ederiz.

Bunlar, C ++ 'da "make" tool ve "makefiles" tarafından ele alınan özelliklerdir. Şimdi, makefile dosyalarının tüm ana yönlerini ve bunların C ++ 'daki uygulamalarını açıklayacağım.

### Make Tool

Make bir [UNIX](https://en.wikipedia.org/wiki/Unix) aracıdır ve bir projenin farklı modüllerinden çalıştırılabilir yapıyı basitleştirmek için bir araç olarak kullanılır. Makefile'da hedef girdiler olarak belirtilen çeşitli kurallar vardır. Make tool tüm bu kuralları okur ve buna göre davranır.

**Örneğin**, bir kural herhangi bir bağımlılık belirtiyorsa, bu durumda make tool derleme amaçlı bu bağımlılığı içerecektir. Make komutu makefile'da modüller oluşturmak veya dosyaları temizlemek için kullanılır. 

**Make'in genel sözdizimi şöyledir:**

```makefile
%make target_label             #target_label makefile'da belirli bir hedeftir
```

*Örneğin*, dosyaları temizlemek için rm komutlarını çalıştırmak istersek şunu yazarız:

```makefile
%make clean                #burada clean, rm komutları için belirtilen bir target_label'dır.
```
### C++ Makefile

Makefile, hedefleri oluşturmak için "make" komutu tarafından kullanılan veya başvurulan bir metin dosyasından başka bir şey değildir. Bir makefile ayrıca her dosya için kaynak düzeyi bağımlılıkları ve yapı sırası bağımlılıkları gibi bilgileri içerir.

Şimdi makefile'ın genel yapısına bakalım.

Bir makefile tipik olarak değişken tanımlamaları(variable declarations) ile başlar ve ardından belirli hedefler oluşturmak için bir dizi hedef girdi gelir. Bu hedefler, .o veya C veya C ++ 'daki diğer yürütülebilir dosyalar ve Java'daki .class dosyaları olabilir.

Ayrıca, hedef etiket tarafından belirtilen bir dizi komutu yürütmek için bir dizi hedef girişimiz olabilir.

**Dolayısıyla genel bir makefile aşağıda gösterildiği gibidir:**

```makefile
# comment
 
target:  dependency1 dependency2 ... dependencyn
      <tab> command
 
# (not: make'in çalışması için komut satırındaki <tab> gereklidir)
```

**Makefile'ın basit bir örneği aşağıda gösterilmiştir.**

```makefile
# myprogram.o ve mylib.lib'den yürütülebilir(executable) myprogram oluşturmak için bir yapı komutu
all:myprogram.o mylib.o
        gcc –o myprogram myprogram.o mylib.o
clean:
        $(RM) myprogram
```

Yukarıdaki makefile'da iki hedef etiket(target labels) belirledik, ilki myprogram ve mylib nesne dosyalarından çalıştırılabilir oluşturmak için "all" etiketidir. İkinci hedef etiket "clean", "myprogram" adındaki tüm dosyaları kaldırır.

**Makefile'ın başka bir varyasyonunu görelim.**

```makefile
# derleyici: C programı için gcc, C ++ için g ++ olarak tanımlayın
  CC = gcc
 
  # derleyici bayrakları(compiler flags):
  #  -g     - bu bayrak çalıştırılabilir dosyaya hata ayıklama bilgisi ekler
  #  -Wall  - bu bayrak çoğu derleyici uyarısını açmak için kullanılır
  CFLAGS  = -g -Wall
 
  # The build target 
  TARGET = myprogram
 
  all: $(TARGET)
 
  $(TARGET): $(TARGET).c
              $(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c
 
  clean:
              $(RM) $(TARGET)
```

Yukarıdaki örnekte gösterildiği gibi, bu makefile'da, kullandığımız derleyici değerini içeren 'CC' değişkenini kullanıyoruz (bu durumda GCC). Başka bir değişken "CFLAGS", kullanacağımız derleyici bayraklarını içerir.

Üçüncü değişken 'TARGET', oluşturmamız gereken çalıştırılabilir dosya için programın adını içerir.

Makefile'ın bu varyasyonunun ölçü avantajı, derleyici, derleyici bayrakları veya çalıştırılabilir program adında bir değişiklik olduğunda sadece kullandığımız değişkenlerin değerlerini değiştirmemiz gerektiğidir.

**Make Ve Makefile Örneği**

Aşağıdaki dosyaları içeren bir program örneğini düşünün:

* **Main.cpp**: Ana sürücü programı
* **Point.h**: Point sınıfı için başlık dosyası
* **Point.cpp**: Point sınıfı için CPP uygulama dosyası
* **Square.h**: Square sınıfı için başlık dosyası
* **Square.cpp**: Square sınıfı için CPP uygulama dosyası

Yukarıda verilen .cpp ve .h dosyalarıyla, .o dosyaları oluşturmak için bu dosyaları ayrı ayrı derlememiz ve ardından bunları main adlı yürütülebilir dosyaya bağlamamız gerekir.

Şimdi bu dosyaları ayrı ayrı derliyoruz.

* **g++ -c Main.cpp**: main.o oluşturur
* **g++ -c Point.cpp**: point.o oluşturur 
* **g++ -c Square.cpp**: square.o oluşturur

Ardından, çalıştırılabilir main'i oluşturmak için nesne dosyalarını birbirine bağlarız.

**g++ -o main Main.o Point.o Square.o**

Daha sonra, programın belirli bölümleri güncellendiğinde dosyalardan hangisini yeniden derlememiz ve yeniden oluşturmamız gerektiğine karar vermemiz gerekiyor. Bunun için, uygulama dosyalarının her biri için çeşitli bağımlılıkları gösteren bir **bağımlılık tablomuz** olacak.

Yukarıdaki dosyalar için bağımlılık tablosu aşağıda verilmiştir.

![](images/dependencyChart.png)

Dolayısıyla, yukarıdaki bağımlılık tablosunda, kökte çalıştırılabilir "main" i görebiliriz. Çalıştırılabilir "main", nesne dosyalarından oluşur; sırasıyla main.cpp, point.cpp ve square.cpp derlenerek oluşturulan main.o, point.o, square.o.

Tüm cpp uygulamaları, yukarıdaki tabloda gösterildiği gibi başlık dosyalarını kullanır. Yukarıda gösterildiği gibi main.cpp, hem point.h hem de square.h'ye başvurur, sürücü programıdır ve point ve square sınıflarını kullanır.

Sonraki dosya point.cpp referanslar point.h. Üçüncü dosya square.cpp, kare çizmek için bir noktaya ihtiyaç duyacağı için square.h ve point.h dosyalarına da başvurur.

Yukarıdaki bağımlılık tablosundan, .cpp dosyası tarafından başvurulan herhangi bir .cpp dosyası veya .h dosyası her değiştiğinde, o .o dosyasını yeniden oluşturmamız gerektiği açıktır. Örneğin, main.cpp değiştiğinde, main.o'yu yeniden oluşturmamız ve main yürütülebilir dosyayı oluşturmak için nesne dosyalarını yeniden bağlamamız gerekir.

Projede az sayıda dosya varsa, yukarıda verdiğimiz tüm açıklamalar sorunsuz bir şekilde çalışacaktır. Proje çok büyük olduğunda ve dosyalar büyük ve çok fazla olduğunda, dosyaları tekrar tekrar oluşturmak zorlaşır.

Thus, we go for make files and we use to make a tool to build the project and generate the executable.

Bu nedenle, dosya oluşturmaya(make file) gidiyoruz ve projeyi oluşturmak ve yürütülebilir dosyayı oluşturmak için bir make tool kullanıyoruz.

We have already seen various parts of a make file. Note that the file should be named “MAKEFILE” or ‘makefile’ and should be placed in the source folder.
Bir make dosyasının çeşitli kısımlarını zaten gördük. Dosyanın "MAKEFILE" veya "makefile" olarak adlandırılması ve kaynak klasöre yerleştirilmesi gerektiğini unutmayın.
Şimdi yukarıdaki örnek için makefile'ı yazacağız.

**Derleyici ve derleyici bayraklarının değerlerini tutacak değişkenleri aşağıda gösterildiği gibi tanımlayacağız.**

```makefile
CC = g++
CFLAGS = -wall -g
```
Daha sonra makefile'mızda ilk hedefi, yani çalıştırılabilir main'i oluşturuyoruz. Yani bağımlılıkları olan bir hedef yazıyoruz.
main: main.o point.o square.o

Bu nedenle, bu hedefi oluşturma komutu
```makefile
<tab>$(CC) $(CFLAGS) –o main main.o point.o square.o
```

```
Not: Yukarıdaki komut aslında g ++ -wall –g –o main main.o point.o square.o 'ya çevrilir.
```
Bir sonraki hedefimiz main.o, point.o, square.o nesne dosyaları oluşturmak olacaktır.

**Şimdi main.o'yu oluşturmak için hedef şu şekilde yazılacak:**
```makefile
Main.o: main.cpp point.h square.h
```

**Bu hedef için komut şudur:**
```makefile
<tab>$(CC) $(CFLAGS) –c main.cpp
```
**Sonraki dosya point.o aşağıdaki komut kullanılarak oluşturulabilir:**
```makefile
<tab>$(CC) $(CFLAGS) –c point.h
```
Yukarıdaki komutta point.cpp'yi atladık. Bunun nedeni, make'in .o dosyalarının .cpp dosyalarından oluşturulduğunu bilmesidir, bu nedenle yalnızca .h (include file) yeterlidir.

**Benzer şekilde, square.o aşağıdaki komutla oluşturulabilir.**
```makefile
<tab>$(CC) $(CFLAGS) –c square.h point.h
```

**Bu örnek için tüm makefile aşağıda gösterildiği gibi görünecektir:**
```makefile
# Makefile for Writing Make Files Example
 
# *****************************************************
# Variables to control Makefile operation
 
CC = g++
CFLAGS = -Wall -g
 
# ****************************************************
# Targets needed to bring the executable up to date
 
main: main.o Point.o Square.o
    $(CC) $(CFLAGS) -o main main.o Point.o Square.o
 
# The main.o target can be written more simply
 
main.o: main.cpp Point.h Square.h
    $(CC) $(CFLAGS) -c main.cpp
 
Point.o: Point.h
 
Square.o: Square.h Point.h
```
Böylece, üç C ++ dosyasını derleyen ve ardından nesne dosyalarından çalıştırılabilir bir main oluşturan eksiksiz bir makefile'a sahip olduğumuzu görüyoruz.

### Makefiles Avantajları

* Büyük projeler söz konusu olduğunda, makefiles kullanmak projeyi sistematik ve verimli bir şekilde temsil etmemize yardımcı olur.
* Makefiles, kaynak kodun okunmasını ve hata ayıklamasını daha kısa ve kolay hale getirir.
*  Makefiles otomatik olarak yalnızca değiştirilen dosyaları derler. Bu nedenle, projenin bazı bölümleri değiştirildiğinde tüm projeyi yeniden oluşturmamız gerekmez.
* Make tool, birden çok dosyayı aynı anda derlememize olanak tanır, böylece tüm dosyalar tek bir adımda derlenebilir.

### Sonuç

Makefiles, yazılım geliştirme için bir nimettir. Bir C ++ makefile kullanarak daha kısa sürede çözüm üretebiliriz. Ayrıca projenin bir bölümü değiştirildiğinde, makefile tüm projeyi yeniden oluşturmak zorunda kalmadan yalnızca o bölümü yeniden derler ve yeniden oluşturur.

C ++ Makefile, projeyi sistematik ve verimli bir şekilde temsil etmemize izin verir, böylece daha okunabilir ve hata ayıklaması daha kolay hale gelir.

### Referans

https://www.softwaretestinghelp.com/cpp-makefile-tutorial/


